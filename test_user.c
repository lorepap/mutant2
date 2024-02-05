#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <asm/types.h>
#include <sys/socket.h>
#include <linux/netlink.h>

#define NETLINK_USER 25
#define MAX_PAYLOAD 1024

#define COMM_BEGIN 1
#define COMM_END 2

struct sockaddr_nl src_addr, dest_addr;
struct nlmsghdr *nlh = NULL;
struct iovec iov;
struct msghdr msg;

void receive_message() {
    char buffer[MAX_PAYLOAD];

    iov.iov_base = buffer;
    iov.iov_len = sizeof(buffer);
    msg.msg_name = (void *)&src_addr;
    msg.msg_namelen = sizeof(src_addr);
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;

    recvmsg(src_addr.nl_pid, &msg, 0);

    // printf("Received message from kernel: %s\n", buffer);
}

void send_message(int flag, int NETLINK_ID, int sock_fd) {
    char message[MAX_PAYLOAD];
    memset(nlh, 0, NLMSG_SPACE(MAX_PAYLOAD));
    if (nlh == NULL) {
        fprintf(stderr, "Error: nlh pointer is NULL\n");
        return;
    }
    
    nlh->nlmsg_len = NLMSG_SPACE(MAX_PAYLOAD);
    nlh->nlmsg_pid = getpid();
    nlh->nlmsg_flags = 0;

    snprintf(message, MAX_PAYLOAD - 1, "%u", flag);

    strcpy(NLMSG_DATA(nlh), message);

    sendto(sock_fd, nlh, nlh->nlmsg_len, 0, (struct sockaddr *)&dest_addr, sizeof(dest_addr));
}

int main() {
    int sock_fd = socket(PF_NETLINK, SOCK_RAW, NETLINK_USER);
    if (sock_fd < 0) {
        perror("Error creating Netlink socket");
        return EXIT_FAILURE;
    }
    printf("Socket created\n");

    memset(&src_addr, 0, sizeof(src_addr));
    src_addr.nl_family = AF_NETLINK;
    src_addr.nl_pid = getpid();

    memset(&dest_addr, 0, sizeof(dest_addr));
    dest_addr.nl_family = AF_NETLINK;
    dest_addr.nl_pid = 0; // For kernel
    dest_addr.nl_groups = 0; // Unicast

    bind(sock_fd, (struct sockaddr *)&src_addr, sizeof(src_addr));

    // Initialization for sending messages
    nlh = (struct nlmsghdr *)malloc(NLMSG_SPACE(MAX_PAYLOAD));

    if (nlh == NULL) {
    fprintf(stderr, "Error: nlh pointer is NULL\n");
    return;
    }
    printf("nlh allocated\n");

    // Set up iov and msg structures
    iov.iov_base = nlh;
    iov.iov_len = nlh->nlmsg_len;
    msg.msg_name = (void *)&dest_addr;
    msg.msg_namelen = sizeof(dest_addr);
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    printf("iov and msg set up\n");

    // Send a message to the kernel
    send_message(COMM_BEGIN, NETLINK_USER, sock_fd);

    // // Receive messages from the kernel
    // while (1) {
    //     receive_message();
    // }

    free(nlh);  // Free allocated memory before exiting
    close(sock_fd);

    return EXIT_SUCCESS;
}
