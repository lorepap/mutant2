#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/device.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Your Name");

#define DEVICE_NAME "shared_device"
#define CLASS_NAME "shared"

#define SHARED_MEMORY_SIZE 256

static struct shared_data {
    char data[SHARED_MEMORY_SIZE];
} *kernel_shared_data;

static int majorNumber;
static struct class* sharedClass = NULL;
static struct device* sharedDevice = NULL;

static int device_open(struct inode*, struct file*);
static ssize_t device_read(struct file*, char*, size_t, loff_t*);
static int device_release(struct inode*, struct file*);

// Work queue handler
static void async_work_handler(struct work_struct *work);
// Declare and initialize a work queue
static DECLARE_WORK(async_work, async_work_handler);

static struct file_operations fops = {
    .open = device_open,
    .read = device_read,
    .release = device_release,
};

static int device_open(struct inode* inode, struct file* file) {
    // Prevent multiple access to the device simultaneously
    if (file->f_flags & O_NONBLOCK)
        return -EBUSY;

    try_module_get(THIS_MODULE);
    return 0;
}

static ssize_t device_read(struct file* file, char* buffer, size_t length, loff_t* offset) {
    int bytesRead = 0;

    // Copy data from kernel_shared_data to user space
    if (copy_to_user(buffer, kernel_shared_data->data, SHARED_MEMORY_SIZE) != 0) {
        return -EFAULT;
    }

    // Return the number of bytes read
    return bytesRead;
}

static int device_release(struct inode* inode, struct file* file) {
    module_put(THIS_MODULE);
    return 0;
}

// Asynchronous work handler
static void async_work_handler(struct work_struct *work) {
    // Simulate writing asynchronous data to shared memory
    snprintf(kernel_shared_data->data, SHARED_MEMORY_SIZE, "Async Data from Kernel");

    // Schedule a read operation for the user space
    schedule_work(&async_work);
}

static int simple_init(void) {
    // Allocate memory for the shared data
    kernel_shared_data = kmalloc(sizeof(struct shared_data), GFP_KERNEL);
    if (!kernel_shared_data) {
        printk(KERN_ALERT "Failed to allocate memory for shared data\n");
        return -ENOMEM;
    }

    // Initialize the character device
    majorNumber = register_chrdev(0, DEVICE_NAME, &fops);
    if (majorNumber < 0) {
        printk(KERN_ALERT "Failed to register a major number\n");
        return majorNumber;
    }

    // Register the device class
    sharedClass = class_create(THIS_MODULE, CLASS_NAME);
    if (IS_ERR(sharedClass)) {
        unregister_chrdev(majorNumber, DEVICE_NAME);
        printk(KERN_ALERT "Failed to register device class\n");
        return PTR_ERR(sharedClass);
    }

    // Create the device
    sharedDevice = device_create(sharedClass, NULL, MKDEV(majorNumber, 0), NULL, DEVICE_NAME);
    if (IS_ERR(sharedDevice)) {
        class_destroy(sharedClass);
        unregister_chrdev(majorNumber, DEVICE_NAME);
        printk(KERN_ALERT "Failed to create the device\n");
        return PTR_ERR(sharedDevice);
    }

    printk(KERN_INFO "Kernel module initialized\n");

     // Initialize the work queue
    INIT_WORK(&async_work, async_work_handler);

    // Schedule the first work execution
    schedule_work(&async_work);

    return 0;
}

static void simple_exit(void) {
    // Destroy the device
    device_destroy(sharedClass, MKDEV(majorNumber, 0));

    // Unregister the device class
    class_unregister(sharedClass);
    class_destroy(sharedClass);

    // Unregister the major number
    unregister_chrdev(majorNumber, DEVICE_NAME);

    // Free the allocated memory
    kfree(kernel_shared_data);

    printk(KERN_INFO "Kernel module exited\n");
}

module_init(simple_init);
module_exit(simple_exit);
