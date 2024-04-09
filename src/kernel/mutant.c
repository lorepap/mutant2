#include <asm-generic/errno.h>
#include <asm-generic/errno-base.h>
#include "mutant.h"

#define NETLINK_USER 25
#define MAX_PAYLOAD 256

// Communication Flags
#define COMM_END 0
#define COMM_BEGIN 1
#define COMM_SELECT_ARM 2
#define TEST 3

// Netlink comm variables
struct sock *nl_sk = NULL;
static u32 socketId = -1;
static u32 selected_proto_id = CUBIC;
static u32 prev_proto_id = CUBIC;
static bool switching_flag = false;
struct mutant_info info;

// Debugging variables
u64 fail_cnt = 0;
u64 success_cnt = 0;

u64 thruput = 0;
u64 loss_rate = 0.0;

// Wrapper struct to call the tcp_congestion_ops of the selected policy
struct tcp_mutant_wrapper {
    struct tcp_congestion_ops *current_ops;
};

static struct tcp_mutant_wrapper mutant_wrapper;
extern struct tcp_congestion_ops cubictcp;
extern struct tcp_congestion_ops tcp_hybla;
extern struct tcp_congestion_ops tcp_bbr_cong_ops;
extern struct tcp_congestion_ops tcp_westwood;
extern struct tcp_congestion_ops tcp_veno;
extern struct tcp_congestion_ops tcp_vegas;
extern struct tcp_congestion_ops tcp_yeah;
extern struct tcp_congestion_ops tcp_cdg;
extern struct tcp_congestion_ops bic;
extern struct tcp_congestion_ops htcp;
extern struct tcp_congestion_ops tcp_highspeed;
extern struct tcp_congestion_ops tcp_illinois;
// extern struct tcp_congestion_ops tcp_pcc_cong_ops;
// extern struct tcp_congestion_ops tcp_base;
// extern struct tcp_congestion_ops tcp_base2;

struct mutant_state {
    struct bictcp *cubic_state;
    struct hybla *hybla_state;
    struct bbr *bbr_state;
    struct westwood *westwood_state;
    struct veno *veno_state;
    struct vegas *vegas_state;
    struct yeah *yeah_state;
    struct cdg *cdg_state;
    struct bic *bic_state;
    struct htcp *htcp_state;
    struct hstcp *highspeed_state;
    struct illinois *illinois_state;
    struct pcc_data *pcc_state;
    // Add more pointers for other congestion control schemes as needed
};

// Global variable to store the state
static struct mutant_state *saved_states;
// Pointe to a general state
static void *crt_state;

// Netlink comm APIs
static void send_msg(char *message, int socketId)
{
    int retryCounter = 0;
    int messageSize;
    int messageSentReponseCode;
    struct sk_buff *socketMessage;
    struct nlmsghdr *reply_nlh = NULL;

    if (socketId == -1)
    {
        printk(KERN_INFO "Message not sent: socket not initialized (-1)");
        return;
    }

    messageSize = strlen(message);

    socketMessage = nlmsg_new(messageSize, 0);

    if (!socketMessage)
    {
        printk(KERN_ERR "Mutant | %s: Failed to allocate new skb | (PID = %d)\n", __FUNCTION__, socketId);
        return;
    }

    reply_nlh = nlmsg_put(socketMessage, 0, 0, NLMSG_DONE, messageSize, 0);

    NETLINK_CB(socketMessage).dst_group = 0; /* not in mcast group */

    strncpy(NLMSG_DATA(reply_nlh), message, messageSize);

    if (nl_sk == NULL)
    {
        printk(KERN_ERR "Mutant | %s: nl_sk is NULL | (PID = %d)\n", __FUNCTION__, socketId);
        return;
    }

    messageSentReponseCode = nlmsg_unicast(nl_sk, socketMessage, socketId);

    if (messageSentReponseCode < 0)
    {
        fail_cnt++;
        // printk(KERN_ERR "Mutant | %s: Error while sending message | (PID = %d) | (Error = %d) | (Fail counter = %d)\n", __FUNCTION__, socketId, messageSentReponseCode, fail_cnt);
    }
    // else
    // {
    //     success_cnt++;
    //     printk("Mutant | %s: Correctly sent message to app layer | PID = %d | Success counter: %d", __FUNCTION__, socketId, success_cnt);
    // }
}

static void start_connection(struct nlmsghdr *nlh)
{
    // success_cnt = 0;
    fail_cnt = 0;
    prev_proto_id = CUBIC;

    // Initialize saved_states
    init_saved_states();

	printk(KERN_INFO "User-kernel communication initialized");
    char message[MAX_PAYLOAD - 1];

    socketId = nlh->nlmsg_pid;

    // Send application layer a message to inform that the connection has started
    snprintf(message, MAX_PAYLOAD - 1, "%u", 0);
    send_msg(message, socketId);
}

static void end_connection(struct nlmsghdr *nlh)
{
    printk(KERN_INFO "User-kernel communication ended");
    char message[MAX_PAYLOAD - 1];

    // Send application layer a message to inform that the connection has ended
    snprintf(message, MAX_PAYLOAD - 1, "%u", -1);
    send_msg(message, socketId);

    // free all the memory allocated for the saved states
    if (saved_states) {
        if (saved_states->cubic_state) {
            kfree(saved_states->cubic_state);
        }
        if (saved_states->hybla_state) {
            kfree(saved_states->hybla_state);
        }
        if (saved_states->bbr_state) {
            kfree(saved_states->bbr_state);
        }
        if (saved_states->westwood_state) {
            kfree(saved_states->westwood_state);
        }
        if (saved_states->veno_state) {
            kfree(saved_states->veno_state);
        }
        if (saved_states->vegas_state) {
            kfree(saved_states->vegas_state);
        }
        if (saved_states->yeah_state) {
            kfree(saved_states->yeah_state);
        }
        if (saved_states->cdg_state) {
            kfree(saved_states->cdg_state);
        }
        if (saved_states->htcp_state) {
            kfree(saved_states->htcp_state);
        }
        if (saved_states->highspeed_state) {
            kfree(saved_states->highspeed_state);
        }
        if (saved_states->illinois_state) {
            kfree(saved_states->illinois_state);
        }
        if (saved_states->pcc_state) {
            kfree(saved_states->pcc_state);
        }
        kfree(saved_states);
    }
    
    // Reset the current_ops to the default
    mutant_wrapper.current_ops = &cubictcp;
    
    socketId = -1;
}


static void receive_msg(struct sk_buff *skb)
{
    struct nlmsghdr *nlh = NULL;

    if (skb == NULL)
    {
        printk(KERN_ERR "Mutant | %s: skb is NULL\n", __FUNCTION__);
        return;
    }

    nlh = (struct nlmsghdr *)skb->data;
	// printk(KERN_INFO "received data");
    switch (nlh->nlmsg_flags)
    {
    case COMM_END:
        printk(KERN_INFO "%s: End connection signal received.", __FUNCTION__);
        end_connection(nlh);
        break;

    case COMM_BEGIN:
        printk(KERN_INFO "%s: Start connection signal received.", __FUNCTION__);
        start_connection(nlh);
        break;

    case COMM_SELECT_ARM:
        if (nlh->nlmsg_seq != selected_proto_id) {
            switching_flag = true;
            prev_proto_id = selected_proto_id;
            selected_proto_id = nlh->nlmsg_seq; 
            printk("%s: received switching signal (id: %d->%d)", __FUNCTION__, prev_proto_id, selected_proto_id);
        }
        break;
    default: // testing
		printk("Test message received!");
        break;
    }
}


static int __init netlink_init(void)
{
    struct netlink_kernel_cfg cfg = {
        .input = receive_msg,
    };

    nl_sk = netlink_kernel_create(&init_net, NETLINK_USER, &cfg);
    if (!nl_sk) {
        printk(KERN_ALERT "Error creating netlink socket.\n");
        return -10;
    }

    printk(KERN_INFO "Netlink socket created successfully.\n");
    return 0;
}

static void __exit netlink_exit(void)
{
    netlink_kernel_release(nl_sk);
    printk(KERN_INFO "Netlink socket released.\n");
}


static void init_saved_states(void) {
    // TODO: The state could be initialized with the init function (see load_state)
    saved_states = kmalloc(sizeof(struct mutant_state), GFP_KERNEL);
    if (!saved_states) {
        pr_err("Failed to allocate memory for saved_states\n");
        return;
    }
    // Initialize pointers to NULL initially
    saved_states->cubic_state = NULL;
    saved_states->hybla_state = NULL;
    saved_states->bbr_state = NULL;
    saved_states->westwood_state = NULL;
    saved_states->veno_state = NULL;
    saved_states->vegas_state = NULL;
    saved_states->yeah_state = NULL;
    saved_states->cdg_state = NULL;
    saved_states->htcp_state = NULL;
    saved_states->highspeed_state = NULL;
    saved_states->illinois_state = NULL;
    saved_states->pcc_state = NULL;
    // Initialize other pointers as needed
}


// FOR DEBUG
static void print_bictcp(struct bictcp *cubic) {
    printk("BiC-TCP State:\n");
    printk("[DEBUG] cnt: %d\n", cubic->cnt);
    printk("[DEBUG] last_max_cwnd: %d\n", cubic->last_max_cwnd);
    printk("[DEBUG] last_cwnd: %d\n", cubic->last_cwnd);
    printk("[DEBUG] last_time: %d\n", cubic->last_time);
    printk("[DEBUG] bic_origin_point: %d\n", cubic->bic_origin_point);
    printk("[DEBUG] bic_K: %d\n", cubic->bic_K);
    printk("[DEBUG] delay_min: %d\n", cubic->delay_min);
    printk("[DEBUG] ack_cnt: %d\n", cubic->ack_cnt);
    printk("[DEBUG] tcp_cwnd: %d\n", cubic->tcp_cwnd);
    printk("[DEBUG] found: %d\n", cubic->found);
    // Print other fields as needed
}

static void print_hybla(struct hybla *hybla) {
    printk("Hybla State:\n");
    printk("[DEBUG] hybla_en: %d\n", hybla->hybla_en);
    printk("[DEBUG] snd_cwnd_cents: %d\n", hybla->snd_cwnd_cents);
    printk("[DEBUG] rho: %d\n", hybla->rho);
    printk("[DEBUG] rho2: %d\n", hybla->rho2);
    printk("[DEBUG] rho_3ls: %d\n", hybla->rho_3ls);
    printk("[DEBUG] rho2_7ls: %d\n", hybla->rho2_7ls);
    printk("[DEBUG] minrtt_us: %d\n", hybla->minrtt_us);
}

static void print_bbr(struct bbr *bbr){
    return;
}

static void print_mutant_state(struct sock *sk) {
    if (selected_proto_id == CUBIC && saved_states->cubic_state) {
        memcpy(saved_states->cubic_state, inet_csk_ca(sk), sizeof(struct bictcp));
        print_bictcp(saved_states->cubic_state);
    }
    else if (selected_proto_id == HYBLA && saved_states->hybla_state) {
        printk("print hybla state\n");
        memcpy(saved_states->hybla_state, inet_csk_ca(sk), sizeof(struct hybla));
        print_hybla(saved_states->hybla_state);
    }
    else return;
    // else if (selected_proto_id == 3) {
    //     print_bbr(saved_states->bbr_state);
    // } else {
    //     printk("BBR state is NULL\n");
    // }

    // Add more conditions for other congestion control schemes as needed
}
////////////////////////////////////////////////


// Function to save the state of a specific congestion control scheme
static void save_state(struct sock *sk) {
    if (!saved_states) {
        pr_err("saved_states not initialized\n");
        return;
    }
    // printk("Saving state of %d", prev_proto_id);
    // Save state of the current congestion control protocol
    struct tcp_sock *tp = tcp_sk(sk);
    switch (prev_proto_id)
    {
        case CUBIC:
            if (saved_states->cubic_state) {
                kfree(saved_states->cubic_state);
            }
            saved_states->cubic_state = kmalloc(sizeof(struct bictcp), GFP_KERNEL);
            if (saved_states->cubic_state) {
                memcpy(saved_states->cubic_state, inet_csk_ca(sk), sizeof(struct bictcp));
                // saved_states->cubic_state->cwnd = tp->snd_cwnd;
            } else {
                pr_err("Failed to allocate memory for cubic_state\n");
            }
            break;
        case HYBLA:
            if (saved_states->hybla_state) {
                kfree(saved_states->hybla_state);
            }
            saved_states->hybla_state = kmalloc(sizeof(struct hybla), GFP_KERNEL);
            if (saved_states->hybla_state) {
                memcpy(saved_states->hybla_state, inet_csk_ca(sk), sizeof(struct hybla));
                // saved_states->hybla_state->cwnd = tp->snd_cwnd;
            } else {
                pr_err("Failed to allocate memory for hybla_state\n");
            }
            break;
        case BBR:
            if (saved_states->bbr_state) {
                kfree(saved_states->bbr_state);
            }
            saved_states->bbr_state = kmalloc(sizeof(struct bbr), GFP_KERNEL);
            if (saved_states->bbr_state) {
                memcpy(saved_states->bbr_state, inet_csk_ca(sk), sizeof(struct bbr));
                // saved_states->bbr_state->cwnd = tp->snd_cwnd;
            } else {
                pr_err("Failed to allocate memory for bbr_state\n");
            }
            break;
        case WESTWOOD:
            if (saved_states->westwood_state) {
                kfree(saved_states->westwood_state);
            }
            saved_states->westwood_state = kmalloc(sizeof(struct westwood), GFP_KERNEL);
            if (saved_states->westwood_state) {
                memcpy(saved_states->westwood_state, inet_csk_ca(sk), sizeof(struct westwood));
                // saved_states->westwood_state->cwnd = tp->snd_cwnd;
            } else {
                pr_err("Failed to allocate memory for westwood_state\n");
            }
            break;
        case VENO:
            if (saved_states->veno_state) {
                kfree(saved_states->veno_state);
            }
            saved_states->veno_state = kmalloc(sizeof(struct veno), GFP_KERNEL);
            if (saved_states->veno_state) {
                memcpy(saved_states->veno_state, inet_csk_ca(sk), sizeof(struct veno));
                // saved_states->veno_state->cwnd = tp->snd_cwnd;
            } else {
                pr_err("Failed to allocate memory for veno_state\n");
            }
            break;
        case VEGAS:
            if (saved_states->vegas_state) {
                kfree(saved_states->vegas_state);
            }
            saved_states->vegas_state = kmalloc(sizeof(struct vegas), GFP_KERNEL);
            if (saved_states->vegas_state) {
                memcpy(saved_states->vegas_state, inet_csk_ca(sk), sizeof(struct vegas));
                // saved_states->vegas_state->cwnd = tp->snd_cwnd;
            } else {
                pr_err("Failed to allocate memory for vegas_state\n");
            }
            break;
        case YEAH:
            if (saved_states->yeah_state) {
                kfree(saved_states->yeah_state);
            }
            saved_states->yeah_state = kmalloc(sizeof(struct yeah), GFP_KERNEL);
            if (saved_states->yeah_state) {
                memcpy(saved_states->yeah_state, inet_csk_ca(sk), sizeof(struct yeah));
                // saved_states->yeah_state->cwnd = tp->snd_cwnd;
            } else {
                pr_err("Failed to allocate memory for yeah_state\n");
            }
            break;
        case CDG:
            if (saved_states->cdg_state) {
                kfree(saved_states->cdg_state);
            }
            saved_states->cdg_state = kmalloc(sizeof(struct cdg), GFP_KERNEL);
            if (saved_states->cdg_state) {
                memcpy(saved_states->cdg_state, inet_csk_ca(sk), sizeof(struct cdg));
                // saved_states->cdg_state->cwnd = tp->snd_cwnd;
            } else {
                pr_err("Failed to allocate memory for cdg_state\n");
            }
            break;
        case BIC:
            if (saved_states->bic_state) {
                kfree(saved_states->bic_state);
            }
            saved_states->bic_state = kmalloc(sizeof(struct bic), GFP_KERNEL);
            if (saved_states->bic_state) {
                memcpy(saved_states->bic_state, inet_csk_ca(sk), sizeof(struct bic));
                // saved_states->bic_state->cwnd = tp->snd_cwnd;
            } else {
                pr_err("Failed to allocate memory for bic_state\n");
            }
            break;
        case HTCP:
            if (saved_states->htcp_state) {
                kfree(saved_states->htcp_state);
            }
            saved_states->htcp_state = kmalloc(sizeof(struct htcp), GFP_KERNEL);
            if (saved_states->htcp_state) {
                memcpy(saved_states->htcp_state, inet_csk_ca(sk), sizeof(struct htcp));
                // saved_states->htcp_state->cwnd = tp->snd_cwnd;
            } else {
                pr_err("Failed to allocate memory for htcp_state\n");
            }
            break;
        case HIGHSPEED:
            if (saved_states->highspeed_state) {
                kfree(saved_states->highspeed_state);
            }
            saved_states->highspeed_state = kmalloc(sizeof(struct hstcp), GFP_KERNEL);
            if (saved_states->highspeed_state) {
                memcpy(saved_states->highspeed_state, inet_csk_ca(sk), sizeof(struct hstcp));
                // saved_states->highspeed_state->cwnd = tp->snd_cwnd;
            } else {
                pr_err("Failed to allocate memory for highspeed_state\n");
            }
            break;
        case ILLINOIS:
            if (saved_states->illinois_state) {
                kfree(saved_states->illinois_state);
            }
            saved_states->illinois_state = kmalloc(sizeof(struct illinois), GFP_KERNEL);
            if (saved_states->illinois_state) {
                memcpy(saved_states->illinois_state, inet_csk_ca(sk), sizeof(struct illinois));
                // saved_states->illinois_state->cwnd = tp->snd_cwnd;
            } else {
                pr_err("Failed to allocate memory for illinois_state\n");
            }
            break;
        // case PCC:
        //     if (saved_states->pcc_state) {
        //         kfree(saved_states->pcc_state);
        //     }
        //     saved_states->pcc_state = kmalloc(sizeof(struct pcc_data), GFP_KERNEL);
        //     if (saved_states->pcc_state) {
        //         memcpy(saved_states->pcc_state, inet_csk_ca(sk), sizeof(struct pcc_data));
        //         // saved_states->pcc_state->cwnd = tp->snd_cwnd;
        //     } else {
        //         pr_err("Failed to allocate memory for pcc_state\n");
        //     }
        //     break;
        default:
            break;
    }
    // printk("[%d] Save cwnd = %d", prev_proto_id, tp->snd_cwnd);
}

// Function to load the state of a specific congestion control scheme
static void load_state(struct sock *sk){
    struct tcp_congestion_ops *cubic;
    struct tcp_congestion_ops *hybla;
    struct tcp_congestion_ops *bbr;
    struct tcp_congestion_ops *westwood;
    struct tcp_congestion_ops *veno;
    struct tcp_congestion_ops *vegas;
    struct tcp_congestion_ops *yeah;
    struct tcp_congestion_ops *cdg;
    struct tcp_congestion_ops *tcp_bic;
    struct tcp_congestion_ops *tcp_htcp;
    struct tcp_congestion_ops *highspeed;
    struct tcp_congestion_ops *illinois;
    // struct tcp_congestion_ops *pcc;
    cubic = &cubictcp;
    hybla = &tcp_hybla;
    bbr = &tcp_bbr_cong_ops;
    westwood = &tcp_westwood;
    veno = &tcp_veno;
    vegas = &tcp_vegas;
    yeah = &tcp_yeah;
    cdg = &tcp_cdg;
    tcp_bic = &bic;
    tcp_htcp = &htcp;
    highspeed = &tcp_highspeed;
    illinois = &tcp_illinois;
    // pcc = &tcp_pcc_cong_ops;

    struct tcp_sock *tp = tcp_sk(sk);

    if (!saved_states) {
        pr_err("saved_states not initialized\n");
        return;
    }

    switch (selected_proto_id)
    {
    case CUBIC:
        if (saved_states->cubic_state) {
            // tp->snd_cwnd = saved_states->cubic_state->cwnd;
            // printk("%s: Loading Cubic state.", __FUNCTION__);
            memcpy(inet_csk_ca(sk), saved_states->cubic_state, sizeof(struct bictcp));
        }
        else{
            // printk("%s: Initializing Cubic state", __FUNCTION__);
            saved_states->cubic_state = kmalloc(sizeof(struct bictcp), GFP_KERNEL);
            memcpy(inet_csk_ca(sk), saved_states->cubic_state, sizeof(struct bictcp));
            cubic->init(sk);
            memcpy(saved_states->cubic_state, inet_csk_ca(sk), sizeof(struct bictcp));
        }
        break;
    case HYBLA:
        if (saved_states->hybla_state) {
            // tp->snd_cwnd = saved_states->hybla_state->cwnd;
            // printk("%s: Loading Hybla state.", __FUNCTION__);
            // printk("%s: printing saved_states->hybla_state", __FUNCTION__);
            // print_hybla(saved_states->hybla_state);
            memcpy(inet_csk_ca(sk), saved_states->hybla_state, sizeof(struct hybla));
            // printk("%s: printing hybla inet_csk_ca(sk)", __FUNCTION__);
            // print_hybla(inet_csk_ca(sk));
        }
        else{
            // printk("%s: Initializing Hybla state", __FUNCTION__);
            saved_states->hybla_state = kmalloc(sizeof(struct hybla), GFP_KERNEL);
            memcpy(inet_csk_ca(sk), saved_states->hybla_state, sizeof(struct hybla));
            hybla->init(sk);
            memcpy(saved_states->hybla_state, inet_csk_ca(sk), sizeof(struct hybla));
        }
        break;
     case BBR:
        if (saved_states->bbr_state) {
            // tp->snd_cwnd = saved_states->bbr_state->cwnd;
            // printk("%s: Loading BBR state.", __FUNCTION__);
            memcpy(inet_csk_ca(sk), saved_states->bbr_state, sizeof(struct bbr));
        }
        else{
            // printk("%s: Initializing BBR state", __FUNCTION__);
            saved_states->bbr_state = kmalloc(sizeof(struct bbr), GFP_KERNEL);
            memcpy(inet_csk_ca(sk), saved_states->bbr_state, sizeof(struct bbr));
            bbr->init(sk);
            memcpy(saved_states->bbr_state, inet_csk_ca(sk), sizeof(struct bbr));
        }
        break;
    case WESTWOOD:
        if (saved_states->westwood_state) {
            // tp->snd_cwnd = saved_states->westwood_state->cwnd;
            // printk("%s: Loading Westwood state.", __FUNCTION__);
            memcpy(inet_csk_ca(sk), saved_states->westwood_state, sizeof(struct westwood));
        }
        else{
            // printk("%s: Initializing Westwood state", __FUNCTION__);
            saved_states->westwood_state = kmalloc(sizeof(struct westwood), GFP_KERNEL);
            memcpy(inet_csk_ca(sk), saved_states->westwood_state, sizeof(struct westwood));
            westwood->init(sk);
            memcpy(saved_states->westwood_state, inet_csk_ca(sk), sizeof(struct westwood));
        }
        break;
    case VENO:
        if (saved_states->veno_state) {
            // tp->snd_cwnd = saved_states->veno_state->cwnd;
            // printk("%s: Loading Veno state.", __FUNCTION__);
            memcpy(inet_csk_ca(sk), saved_states->veno_state, sizeof(struct veno));
        }
        else{
            // printk("%s: Initializing Veno state", __FUNCTION__);
            saved_states->veno_state = kmalloc(sizeof(struct veno), GFP_KERNEL);
            memcpy(inet_csk_ca(sk), saved_states->veno_state, sizeof(struct veno));
            veno->init(sk);
            memcpy(saved_states->veno_state, inet_csk_ca(sk), sizeof(struct veno));
        }
        break;
    case VEGAS:
        if (saved_states->vegas_state) {
            // tp->snd_cwnd = saved_states->vegas_state->cwnd;
            // printk("%s: Loading Vegas state.", __FUNCTION__);
            memcpy(inet_csk_ca(sk), saved_states->vegas_state, sizeof(struct vegas));
        }
        else{
            // printk("%s: Initializing Vegas state", __FUNCTION__);
            saved_states->vegas_state = kmalloc(sizeof(struct vegas), GFP_KERNEL);
            memcpy(inet_csk_ca(sk), saved_states->vegas_state, sizeof(struct vegas));
            vegas->init(sk);
            memcpy(saved_states->vegas_state, inet_csk_ca(sk), sizeof(struct vegas));
        }
        break;
    case YEAH:
        if (saved_states->yeah_state) {
            // tp->snd_cwnd = saved_states->yeah_state->cwnd;
            // printk("%s: Loading Yeah state.", __FUNCTION__);
            memcpy(inet_csk_ca(sk), saved_states->yeah_state, sizeof(struct yeah));
        }
        else{
            // printk("%s: Initializing Yeah state", __FUNCTION__);
            saved_states->yeah_state = kmalloc(sizeof(struct yeah), GFP_KERNEL);
            memcpy(inet_csk_ca(sk), saved_states->yeah_state, sizeof(struct yeah));
            yeah->init(sk);
            memcpy(saved_states->yeah_state, inet_csk_ca(sk), sizeof(struct yeah));
        }
        break;
    case CDG:
        if (saved_states->cdg_state) {
            // tp->snd_cwnd = saved_states->cdg_state->cwnd;
            // printk("%s: Loading CDG state.", __FUNCTION__);
            memcpy(inet_csk_ca(sk), saved_states->cdg_state, sizeof(struct cdg));
        }
        else{
            // printk("%s: Initializing CDG state", __FUNCTION__);
            saved_states->cdg_state = kmalloc(sizeof(struct cdg), GFP_KERNEL);
            memcpy(inet_csk_ca(sk), saved_states->cdg_state, sizeof(struct cdg));
            cdg->init(sk);
            memcpy(saved_states->cdg_state, inet_csk_ca(sk), sizeof(struct cdg));
        }
        break;
    case BIC:
        if (saved_states->bic_state) {
            // tp->snd_cwnd = saved_states->bic_state->cwnd;
            // printk("%s: Loading BIC state.", __FUNCTION__);
            memcpy(inet_csk_ca(sk), saved_states->bic_state, sizeof(struct bic));
        }
        else{
            // printk("%s: Initializing BIC state", __FUNCTION__);
            saved_states->bic_state = kmalloc(sizeof(struct bic), GFP_KERNEL);
            memcpy(inet_csk_ca(sk), saved_states->bic_state, sizeof(struct bic));
            tcp_bic->init(sk);
            memcpy(saved_states->bic_state, inet_csk_ca(sk), sizeof(struct bic));
        }
        break;
    case HTCP:
        if (saved_states->htcp_state) {
            // tp->snd_cwnd = saved_states->htcp_state->cwnd;
            // printk("%s: Loading HTCP state.", __FUNCTION__);
            memcpy(inet_csk_ca(sk), saved_states->htcp_state, sizeof(struct htcp));
        }
        else{
            // printk("%s: Initializing HTCP state", __FUNCTION__);
            saved_states->htcp_state = kmalloc(sizeof(struct htcp), GFP_KERNEL);
            memcpy(inet_csk_ca(sk), saved_states->htcp_state, sizeof(struct htcp));
            tcp_htcp->init(sk);
            memcpy(saved_states->htcp_state, inet_csk_ca(sk), sizeof(struct htcp));
        }
        break;
    case HIGHSPEED:
        if (saved_states->highspeed_state) {
            // tp->snd_cwnd = saved_states->highspeed_state->cwnd;
            // printk("%s: Loading Highspeed state.", __FUNCTION__);
            memcpy(inet_csk_ca(sk), saved_states->highspeed_state, sizeof(struct hstcp));
        }
        else{
            // printk("%s: Initializing Highspeed state", __FUNCTION__);
            saved_states->highspeed_state = kmalloc(sizeof(struct hstcp), GFP_KERNEL);
            memcpy(inet_csk_ca(sk), saved_states->highspeed_state, sizeof(struct hstcp));
            highspeed->init(sk);
            memcpy(saved_states->highspeed_state, inet_csk_ca(sk), sizeof(struct hstcp));
        }
        break;
    case ILLINOIS:
        if (saved_states->illinois_state) {
            // tp->snd_cwnd = saved_states->illinois_state->cwnd;
            // printk("%s: Loading Illinois state.", __FUNCTION__);
            memcpy(inet_csk_ca(sk), saved_states->illinois_state, sizeof(struct illinois));
        }
        else{
            // printk("%s: Initializing Illinois state", __FUNCTION__);
            saved_states->illinois_state = kmalloc(sizeof(struct illinois), GFP_KERNEL);
            memcpy(inet_csk_ca(sk), saved_states->illinois_state, sizeof(struct illinois));
            illinois->init(sk);
            memcpy(saved_states->illinois_state, inet_csk_ca(sk), sizeof(struct illinois));
        }
        break;
    // case PCC:
    //     if (saved_states->pcc_state) {
    //         // tp->snd_cwnd = saved_states->pcc_state->cwnd;
    //         // printk("%s: Loading PCC state.", __FUNCTION__);
    //         memcpy(inet_csk_ca(sk), saved_states->pcc_state, sizeof(struct pcc_data));
    //     }
    //     else{
    //         // printk("%s: Initializing PCC state", __FUNCTION__);
    //         saved_states->pcc_state = kmalloc(sizeof(struct pcc_data), GFP_KERNEL);
    //         memcpy(inet_csk_ca(sk), saved_states->pcc_state, sizeof(struct pcc_data));
    //         pcc->init(sk);
    //         memcpy(saved_states->pcc_state, inet_csk_ca(sk), sizeof(struct pcc_data));
    //     }
        break;
    default:
        break;
    }
    // printk("[%d] Loading cwnd = %d", selected_proto_id, tp->snd_cwnd);
}

// Swicthing congestion control function
void mutant_switch_congestion_control(void) {

    switch (selected_proto_id)
    {
    case CUBIC:
        // printk(KERN_INFO "Switching to Cubic (ID: %d)", selected_proto_id);
        mutant_wrapper.current_ops = &cubictcp;
        break;
    case HYBLA:
        // printk(KERN_INFO "Switching to Hybla (ID: %d)", selected_proto_id);
        mutant_wrapper.current_ops = &tcp_hybla;
        break;
    case BBR:
        // printk(KERN_INFO "Switching to BBR (ID: %d)", selected_proto_id);
        mutant_wrapper.current_ops = &tcp_bbr_cong_ops;
        break;
    case WESTWOOD:
        // printk(KERN_INFO "Switching to Westwood (ID: %d)", selected_proto_id);
        mutant_wrapper.current_ops = &tcp_westwood;
        break;
    case VENO:
        // printk(KERN_INFO "Switching to Veno (ID: %d)", selected_proto_id);
        mutant_wrapper.current_ops = &tcp_veno;
        break;
    case VEGAS:
        // printk(KERN_INFO "Switching to Vegas (ID: %d)", selected_proto_id);
        mutant_wrapper.current_ops = &tcp_vegas;
        break;
    case YEAH:
        // printk(KERN_INFO "Switching to Yeah (ID: %d)", selected_proto_id);
        mutant_wrapper.current_ops = &tcp_yeah;
        break;
    case CDG:
        // printk(KERN_INFO "Switching to CDG (ID: %d)", selected_proto_id);
        mutant_wrapper.current_ops = &tcp_cdg;
        break;
    case BIC:
        // printk(KERN_INFO "Switching to BIC (ID: %d)", selected_proto_id);
        mutant_wrapper.current_ops = &bic;
        break;
    case HTCP:
        // printk(KERN_INFO "Switching to HTCP (ID: %d)", selected_proto_id);
        mutant_wrapper.current_ops = &htcp;
        break;
    case HIGHSPEED:
        // printk(KERN_INFO "Switching to Highspeed (ID: %d)", selected_proto_id);
        mutant_wrapper.current_ops = &tcp_highspeed;
        break;
    case ILLINOIS:
        // printk(KERN_INFO "Switching to Illinois (ID: %d)", selected_proto_id);
        mutant_wrapper.current_ops = &tcp_illinois;
        break;
    // case PCC:
    //     // printk(KERN_INFO "Switching to PCC (ID: %d)", selected_proto_id);
    //     mutant_wrapper.current_ops = &tcp_pcc_cong_ops;
    //     break;
    // case BASE:
    //     // printk(KERN_INFO "Switching to Base (ID: %d)", selected_proto_id);
    //     mutant_wrapper.current_ops = &tcp_base;
    //     break;
    // case BASE2:
    //     // printk(KERN_INFO "Switching to Base2 (ID: %d)", selected_proto_id);
    //     mutant_wrapper.current_ops = &tcp_base2;
    //     break;
    default:
        // printk(KERN_INFO "Switching to default (Cubic)");
        mutant_wrapper.current_ops = &cubictcp;
        break;
    }
}

static void send_info(struct mutant_info *info) {
    char msg[MAX_PAYLOAD - 1];

    // printk('[DEBUG] Thruput:', info->thruput);

    snprintf(msg, MAX_PAYLOAD - 1,
    "%u;%u;%u;%u;%u;%u;%u;%u;%u;%u;%u;%u;%u;%u;%u;%u;%u",
    info->now, info->snd_cwnd, info->rtt_us, info->srtt_us, info->mdev_us, 
    info->min_rtt, info->advmss, info->delivered, info->lost_out, 
    info->packets_out, info->retrans_out, info->rate, info->prev_proto_id, 
    info->selected_proto_id, info->thruput, info->loss_rate);
    // printk("Sending info: %s", msg);
    send_msg(msg, socketId);
}

static void send_net_params(struct tcp_sock *tp, struct sock *sk, int socketId)
{
    u32 rate = READ_ONCE(tp->rate_delivered);
    u32 intv = READ_ONCE(tp->rate_interval_us);

    if (tp->packets_out + tp->retrans_out > 0) 
        loss_rate = ((u64) tp->lost_out * 100) / (tp->packets_out + tp->retrans_out);
    
    if (rate && intv) {
        thruput = (u64)rate * tp->mss_cache * USEC_PER_SEC * 8; // USEC_PER_SEC=1e6; 8 to convert to bits (MSS is in bytes); throughput is in bps
        do_div(thruput, intv);
    }
    if (rate==0) {
        thruput = 0;
    }

    info.now = tcp_jiffies32;
    info.snd_cwnd = tp->snd_cwnd;
    info.rtt_us = tp->rack.rtt_us;
    info.srtt_us = tp->srtt_us;
    info.mdev_us = tp->mdev_us;
    info.min_rtt = tcp_min_rtt(tp);
    info.advmss = tp->advmss;
    info.delivered = tp->delivered;
    info.lost_out = tp->lost_out;
    info.packets_out = tp->packets_out;
    info.retrans_out = tp->retrans_out;
    info.rate = rate; 
    info.prev_proto_id = prev_proto_id;  
    info.selected_proto_id = selected_proto_id;  
    info.thruput = thruput;  
    info.loss_rate = loss_rate;

    // printk("Thruput = %u", info.thruput);

    // Send feature values (rtt min >= 10ms)
    if (info.rtt_us>10000)
        send_info(&info);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


static void mutant_tcp_init(struct sock *sk) {
    // Initialize the selected protocol
    if (mutant_wrapper.current_ops->init){
        mutant_wrapper.current_ops->init(sk);
        // printk("Mutant %s: init %s", __FUNCTION__, mutant_wrapper.current_ops->name);
    }
}

static void mutant_tcp_cong_avoid(struct sock *sk, u32 ack, u32 acked) {
    if (mutant_wrapper.current_ops && mutant_wrapper.current_ops->cong_avoid){
        mutant_wrapper.current_ops->cong_avoid(sk, ack, acked);
        // printk("Mutant %s: cong_avoid %s", __FUNCTION__, mutant_wrapper.current_ops->name);
    }
    // Handle null value
    if (sk == NULL) {
        printk("Mutant %s: sk is NULL", __FUNCTION__);
        return;
    }
}

static u32 mutant_tcp_ssthresh(struct sock *sk) {
    if (mutant_wrapper.current_ops && mutant_wrapper.current_ops->ssthresh)
        return mutant_wrapper.current_ops->ssthresh(sk);

    return TCP_INFINITE_SSTHRESH; // Default value in case of no function
}

static void mutant_tcp_set_state(struct sock *sk, u8 new_state) {
    if (mutant_wrapper.current_ops && mutant_wrapper.current_ops->set_state)
        mutant_wrapper.current_ops->set_state(sk, new_state);
}

static u32 mutant_tcp_undo_cwnd(struct sock *sk) {
    if (mutant_wrapper.current_ops && mutant_wrapper.current_ops->undo_cwnd)
        return mutant_wrapper.current_ops->undo_cwnd(sk);

    return tcp_sk(sk)->snd_cwnd; // Default behavior
}

static void mutant_tcp_cwnd_event(struct sock *sk, enum tcp_ca_event event) {
    if (mutant_wrapper.current_ops && mutant_wrapper.current_ops->cwnd_event){
        mutant_wrapper.current_ops->cwnd_event(sk, event);
        // printk("Mutant %s: cwnd_event %s", __FUNCTION__, mutant_wrapper.current_ops->name);
    }
}

static void mutant_tcp_pkts_acked(struct sock *sk, const struct ack_sample *sample) {
    if (mutant_wrapper.current_ops && mutant_wrapper.current_ops->pkts_acked){
        mutant_wrapper.current_ops->pkts_acked(sk, sample);
        // printk("Mutant %s: acked %s", __FUNCTION__, mutant_wrapper.current_ops->name);
    }
    // Send network features every ack (for now)    
    struct tcp_sock *tp = tcp_sk(sk);
    
    if (socketId == -1) {
        return;
    }
    
    // Switching operation
    if (switching_flag) {
        // printk("%s: Switching flag ON", __FUNCTION__);
        save_state(sk);
        mutant_switch_congestion_control();
        load_state(sk);
        switching_flag = false;
    }
    
    send_net_params(tp, sk, socketId);
}

static void mutant_tcp_ack_event(struct sock *sk, u32 flags) {
    if (mutant_wrapper.current_ops && mutant_wrapper.current_ops->in_ack_event)
        mutant_wrapper.current_ops->in_ack_event(sk, flags);
}

static u32 mutant_tcp_cong_control(struct sock *sk, const struct rate_sample *rs, u32 ack, u32 acked, int flag) {
    if (mutant_wrapper.current_ops && mutant_wrapper.current_ops->cong_control){
        mutant_wrapper.current_ops->cong_control(sk, rs);
        // printk("Mutant %s: cong_control %s", __FUNCTION__, mutant_wrapper.current_ops->name);
        return 0;
    }
    // printk("Mutant %s: cong_control %s", __FUNCTION__, mutant_wrapper.current_ops->name);
    return 1;
}

static u32 mutant_tcp_sndbuf_expand(struct sock *sk) {
    if (mutant_wrapper.current_ops && mutant_wrapper.current_ops->sndbuf_expand){
        return mutant_wrapper.current_ops->sndbuf_expand(sk);
        // printk("Mutant %s: sndbuf_expand %s", __FUNCTION__, mutant_wrapper.current_ops->name);
    }
    // printk("Mutant %s: sndbuf_expand %s", __FUNCTION__, mutant_wrapper.current_ops->name);
    return 2;
}

static u32 mutant_tcp_min_tso_segs(struct sock *sk) {
    if (mutant_wrapper.current_ops && mutant_wrapper.current_ops->min_tso_segs){
        return mutant_wrapper.current_ops->min_tso_segs(sk);
        // printk("Mutant %s: min_tso_segs %s", __FUNCTION__, mutant_wrapper.current_ops->name);
    }
    // printk("Mutant %s: min_tso_segs %s", __FUNCTION__, mutant_wrapper.current_ops->name);
    return 2;
}

static size_t mutant_tcp_get_info(struct sock *sk, u32 ext, int *attr,
			   union tcp_cc_info *info) {
    if (mutant_wrapper.current_ops && mutant_wrapper.current_ops->get_info)
        return mutant_wrapper.current_ops->get_info(sk, ext, attr, info);
    else
        return 0; // or some other default value
}

static void mutant_tcp_release(struct sock *sk) {
    if (mutant_wrapper.current_ops && mutant_wrapper.current_ops->release)
        mutant_wrapper.current_ops->release(sk);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


static struct tcp_congestion_ops mutant_cong_ops __read_mostly = {
    .flags		= TCP_CONG_NON_RESTRICTED,
    .init       = mutant_tcp_init,
    .ssthresh   = mutant_tcp_ssthresh,
    .cong_avoid = mutant_tcp_cong_avoid,
    .set_state  = mutant_tcp_set_state,
    .undo_cwnd  = mutant_tcp_undo_cwnd,
    .cwnd_event = mutant_tcp_cwnd_event,
    .pkts_acked = mutant_tcp_pkts_acked,
    .in_ack_event	= mutant_tcp_ack_event,
    .sndbuf_expand = mutant_tcp_sndbuf_expand,
    .min_tso_segs = mutant_tcp_min_tso_segs,
    .get_info   = mutant_tcp_get_info,
    .mutant_tcp_cong_control = mutant_tcp_cong_control,
    .release = mutant_tcp_release,
    .owner      = THIS_MODULE,
    .name       = "mutant",
};


static int __init mutant_tcp_module_init(void) {
    // Netlink init
    if (netlink_init() < 0) {
        pr_err("Netlink could not be initialized\n");
        return -EINVAL;
    }
    
    // Initialize saved_states
    init_saved_states();

    // Initial setup or default congestion control selection
    mutant_wrapper.current_ops = &cubictcp;

    // Register the custom congestion control
    if (tcp_register_congestion_control(&mutant_cong_ops) < 0) {
        pr_err("Mutant congestion control could not be registered\n");
        return -EINVAL;
    }

    return 0;
}

static void __exit mutant_tcp_module_exit(void) {
    netlink_exit();
    tcp_unregister_congestion_control(&mutant_cong_ops);
}

module_init(mutant_tcp_module_init);
module_exit(mutant_tcp_module_exit);
MODULE_AUTHOR("Lorenzo Pappone");
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Mutant");
MODULE_VERSION("1.0");