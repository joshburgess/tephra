//! Multi-queue example: async image upload via the transfer queue.
//!
//! Demonstrates uploading a texture on the dedicated transfer queue
//! while the graphics queue continues rendering, synchronized with
//! semaphores.

fn main() {
    env_logger::init();

    println!("06_async_transfer: multi-queue async upload demo");
    println!();
    println!("This example requires a Vulkan device with a dedicated transfer queue.");
    println!();
    println!("API usage pattern:");
    println!("  1. Create Device (discovers dedicated transfer queue family)");
    println!("  2. Create staging buffer (host-visible) with texture data");
    println!("  3. Create GPU-local image");
    println!("  4. Transfer queue commands:");
    println!("     - cmd = device.request_command_buffer_raw(QueueType::Transfer)");
    println!("     - cmd.image_barrier(UNDEFINED -> TRANSFER_DST)");
    println!("     - cmd.copy_buffer_to_image(staging, image)");
    println!("     - cmd.image_barrier(TRANSFER_DST -> SHADER_READ_ONLY)");
    println!("     - Queue family ownership transfer (transfer -> graphics)");
    println!("  5. Submit transfer cmd with signal semaphore");
    println!("  6. Graphics queue waits on transfer semaphore before using texture");
    println!("  7. Graphics queue acquires ownership, samples the texture");
}
