//! Shader variant/template system for managing `#define`-based shader permutations.
//!
//! Enables defining shader templates with compile-time defines that produce
//! different SPIR-V variants. Each unique combination of defines produces a
//! distinct shader module, cached by its define set hash.
//!
//! # Usage
//!
//! ```ignore
//! let mut registry = ShaderVariantRegistry::new();
//!
//! // Register a shader template
//! let template = ShaderTemplate::new("shaders/pbr.frag.glsl")
//!     .define_bool("HAS_NORMAL_MAP")
//!     .define_bool("HAS_EMISSIVE")
//!     .define_int("MAX_LIGHTS", 4);
//!
//! let template_id = registry.register_template(template);
//!
//! // Request a specific variant
//! let mut defines = DefineSet::new();
//! defines.set_bool("HAS_NORMAL_MAP", true);
//! defines.set_int("MAX_LIGHTS", 16);
//!
//! let variant_key = registry.request_variant(template_id, defines);
//! // The variant can be compiled in the background and cached
//! ```

use std::path::{Path, PathBuf};

use rustc_hash::FxHashMap;

/// A named define with a value.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DefineValue {
    /// Boolean define (present or absent).
    Bool(bool),
    /// Integer define.
    Int(i64),
    /// Float define (stored as bits for hashing).
    Float(u64),
    /// String define.
    String(String),
}

impl DefineValue {
    /// Create a float define value.
    pub fn float(value: f64) -> Self {
        Self::Float(value.to_bits())
    }

    /// Format as a GLSL `#define` value string.
    pub fn to_define_string(&self) -> Option<String> {
        match self {
            DefineValue::Bool(true) => Some("1".to_string()),
            DefineValue::Bool(false) => None, // not defined
            DefineValue::Int(v) => Some(v.to_string()),
            DefineValue::Float(bits) => Some(format!("{}", f64::from_bits(*bits))),
            DefineValue::String(s) => Some(s.clone()),
        }
    }
}

/// A set of defines for a shader variant.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct DefineSet {
    defines: Vec<(String, DefineValue)>,
}

impl DefineSet {
    /// Create an empty define set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a boolean define (true = defined as 1, false = not defined).
    pub fn set_bool(&mut self, name: &str, value: bool) -> &mut Self {
        self.set(name, DefineValue::Bool(value));
        self
    }

    /// Set an integer define.
    pub fn set_int(&mut self, name: &str, value: i64) -> &mut Self {
        self.set(name, DefineValue::Int(value));
        self
    }

    /// Set a float define.
    pub fn set_float(&mut self, name: &str, value: f64) -> &mut Self {
        self.set(name, DefineValue::float(value));
        self
    }

    /// Set a string define.
    pub fn set_string(&mut self, name: &str, value: &str) -> &mut Self {
        self.set(name, DefineValue::String(value.to_string()));
        self
    }

    /// Set a define with an explicit value.
    pub fn set(&mut self, name: &str, value: DefineValue) {
        if let Some(entry) = self.defines.iter_mut().find(|(n, _)| n == name) {
            entry.1 = value;
        } else {
            self.defines.push((name.to_string(), value));
        }
    }

    /// Get a define's value by name.
    pub fn get(&self, name: &str) -> Option<&DefineValue> {
        self.defines.iter().find(|(n, _)| n == name).map(|(_, v)| v)
    }

    /// Iterate over all defines.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &DefineValue)> {
        self.defines.iter().map(|(n, v)| (n.as_str(), v))
    }

    /// Generate the `#define` preamble for GLSL preprocessing.
    pub fn to_preamble(&self) -> String {
        let mut preamble = String::new();
        for (name, value) in &self.defines {
            if let Some(val_str) = value.to_define_string() {
                preamble.push_str(&format!("#define {} {}\n", name, val_str));
            }
        }
        preamble
    }

    /// Compute a hash of this define set for caching.
    pub fn hash_key(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = rustc_hash::FxHasher::default();
        self.defines.hash(&mut hasher);
        hasher.finish()
    }
}

/// A shader template that can produce variants via defines.
#[derive(Debug, Clone)]
pub struct ShaderTemplate {
    /// Path to the source shader file (GLSL or HLSL).
    pub source_path: PathBuf,
    /// Available defines with their default values.
    pub available_defines: Vec<(String, DefineValue)>,
    /// The shader stage.
    pub stage: ash::vk::ShaderStageFlags,
}

impl ShaderTemplate {
    /// Create a new shader template from a source path.
    pub fn new(path: impl AsRef<Path>, stage: ash::vk::ShaderStageFlags) -> Self {
        Self {
            source_path: path.as_ref().to_path_buf(),
            available_defines: Vec::new(),
            stage,
        }
    }

    /// Add a boolean define option to this template.
    pub fn define_bool(mut self, name: &str) -> Self {
        self.available_defines
            .push((name.to_string(), DefineValue::Bool(false)));
        self
    }

    /// Add an integer define option with a default value.
    pub fn define_int(mut self, name: &str, default: i64) -> Self {
        self.available_defines
            .push((name.to_string(), DefineValue::Int(default)));
        self
    }

    /// Add a string define option with a default value.
    pub fn define_string(mut self, name: &str, default: &str) -> Self {
        self.available_defines
            .push((name.to_string(), DefineValue::String(default.to_string())));
        self
    }

    /// Get the default define set for this template.
    pub fn default_defines(&self) -> DefineSet {
        let mut set = DefineSet::new();
        for (name, value) in &self.available_defines {
            set.set(name, value.clone());
        }
        set
    }
}

/// Unique identifier for a registered shader template.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShaderTemplateId(pub u32);

/// Unique identifier for a specific shader variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShaderVariantKey {
    /// Which template this variant belongs to.
    pub template: ShaderTemplateId,
    /// Hash of the define set.
    pub define_hash: u64,
}

/// Registry for shader templates and their compiled variants.
pub struct ShaderVariantRegistry {
    templates: Vec<ShaderTemplate>,
    /// Maps (template_id, define_hash) -> compiled SPIR-V.
    compiled_variants: FxHashMap<ShaderVariantKey, Vec<u32>>,
}

impl ShaderVariantRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            templates: Vec::new(),
            compiled_variants: FxHashMap::default(),
        }
    }

    /// Register a shader template and return its ID.
    pub fn register_template(&mut self, template: ShaderTemplate) -> ShaderTemplateId {
        let id = ShaderTemplateId(self.templates.len() as u32);
        self.templates.push(template);
        id
    }

    /// Get a registered template by ID.
    pub fn template(&self, id: ShaderTemplateId) -> Option<&ShaderTemplate> {
        self.templates.get(id.0 as usize)
    }

    /// Request a variant key for a template with given defines.
    ///
    /// This doesn't compile the variant — it just computes the key.
    /// Use [`is_compiled`](Self::is_compiled) to check if compilation is needed.
    pub fn variant_key(&self, template: ShaderTemplateId, defines: &DefineSet) -> ShaderVariantKey {
        ShaderVariantKey {
            template,
            define_hash: defines.hash_key(),
        }
    }

    /// Check if a variant has been compiled.
    pub fn is_compiled(&self, key: &ShaderVariantKey) -> bool {
        self.compiled_variants.contains_key(key)
    }

    /// Store a compiled SPIR-V variant.
    pub fn store_compiled(&mut self, key: ShaderVariantKey, spirv: Vec<u32>) {
        self.compiled_variants.insert(key, spirv);
    }

    /// Get a compiled variant's SPIR-V.
    pub fn get_compiled(&self, key: &ShaderVariantKey) -> Option<&[u32]> {
        self.compiled_variants.get(key).map(|v| v.as_slice())
    }

    /// Remove all compiled variants for a template (e.g., after source change).
    pub fn invalidate_template(&mut self, template: ShaderTemplateId) {
        self.compiled_variants.retain(|k, _| k.template != template);
    }

    /// Remove all compiled variants.
    pub fn invalidate_all(&mut self) {
        self.compiled_variants.clear();
    }

    /// Total number of compiled variants across all templates.
    pub fn compiled_count(&self) -> usize {
        self.compiled_variants.len()
    }

    /// Total number of registered templates.
    pub fn template_count(&self) -> usize {
        self.templates.len()
    }
}

impl Default for ShaderVariantRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn define_set_preamble() {
        let mut defines = DefineSet::new();
        defines.set_bool("USE_NORMAL_MAP", true);
        defines.set_int("MAX_LIGHTS", 16);
        defines.set_bool("USE_EMISSIVE", false);

        let preamble = defines.to_preamble();
        assert!(preamble.contains("#define USE_NORMAL_MAP 1"));
        assert!(preamble.contains("#define MAX_LIGHTS 16"));
        assert!(!preamble.contains("USE_EMISSIVE"));
    }

    #[test]
    fn define_set_hash_stability() {
        let mut d1 = DefineSet::new();
        d1.set_bool("A", true);
        d1.set_int("B", 42);

        let mut d2 = DefineSet::new();
        d2.set_bool("A", true);
        d2.set_int("B", 42);

        assert_eq!(d1.hash_key(), d2.hash_key());
    }

    #[test]
    fn different_defines_different_hash() {
        let mut d1 = DefineSet::new();
        d1.set_int("B", 42);

        let mut d2 = DefineSet::new();
        d2.set_int("B", 43);

        assert_ne!(d1.hash_key(), d2.hash_key());
    }

    #[test]
    fn variant_registry() {
        let mut reg = ShaderVariantRegistry::new();

        let template = ShaderTemplate::new("test.frag.glsl", ash::vk::ShaderStageFlags::FRAGMENT)
            .define_bool("USE_ALPHA");

        let id = reg.register_template(template);
        assert_eq!(reg.template_count(), 1);

        let mut defines = DefineSet::new();
        defines.set_bool("USE_ALPHA", true);

        let key = reg.variant_key(id, &defines);
        assert!(!reg.is_compiled(&key));

        reg.store_compiled(key, vec![0x07230203]); // SPIR-V magic
        assert!(reg.is_compiled(&key));
        assert_eq!(reg.compiled_count(), 1);

        reg.invalidate_template(id);
        assert!(!reg.is_compiled(&key));
        assert_eq!(reg.compiled_count(), 0);
    }
}
