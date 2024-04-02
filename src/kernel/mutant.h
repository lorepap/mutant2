#ifndef MUTANT_H
#define MUTANT_H

#include <linux/netlink.h>
#include <net/tcp.h>

// Include the user-defined TCP algorithms
#include "protocol/base.h"
#include "protocol/base2.h"


#define MAX_PAYLOAD 256

// Netlink communication flags
#define COMM_END 0
#define COMM_BEGIN 1
#define COMM_SELECT_ARM 2
#define TEST 3

#define CUBIC 0
#define HYBLA 1
#define BBR 2
#define WESTWOOD 3
#define VENO 4
#define VEGAS 5
#define YEAH 6
#define CDG  7
#define BIC 8
#define HTCP 9
#define HIGHSPEED 10
#define ILLINOIS 11
#define PCC 12
#define BASE 13
#define BASE2 14


struct mutant_info {
    u32 now;
    u32 snd_cwnd;
    u32 rtt_us;
    u32 srtt_us;
    u32 mdev_us;
    u32 min_rtt;
    u32 advmss;
    u32 delivered;
    u32 lost_out;
    u32 packets_out;
    u32 retrans_out;
    u64 rate;
    u32 prev_proto_id;
    u32 selected_proto_id;
    u64 thruput;
    u32 loss_rate;
};


// Struct for saving state of TCP congestion control
// Cubic variables
struct bictcp {
	u32	cnt;		/* increase cwnd by 1 after ACKs */
	u32	last_max_cwnd;	/* last maximum snd_cwnd */
	u32	last_cwnd;	/* the last snd_cwnd */
	u32	last_time;	/* time when updated last_cwnd */
	u32	bic_origin_point;/* origin point of bic function */
	u32	bic_K;		/* time to origin point
				   from the beginning of the current epoch */
	u32	delay_min;	/* min delay (msec << 3) */
	u32	epoch_start;	/* beginning of an epoch */
	u32	ack_cnt;	/* number of acks */
	u32	tcp_cwnd;	/* estimated tcp cwnd */
	u16	unused;
	u8	sample_cnt;	/* number of samples to decide curr_rtt */
	u8	found;		/* the exit point is found? */
	u32	round_start;	/* beginning of each round */
	u32	end_seq;	/* end_seq of the round */
	u32	last_ack;	/* last time when the ACK spacing is close */
	u32	curr_rtt;	/* the minimum rtt of current round */
	
};

/* Tcp Hybla structure. */
struct hybla {
	bool  hybla_en;
	u32   snd_cwnd_cents; /* Keeps increment values when it is <1, <<7 */
	u32   rho;	      /* Rho parameter, integer part  */
	u32   rho2;	      /* Rho * Rho, integer part */
	u32   rho_3ls;	      /* Rho parameter, <<3 */
	u32   rho2_7ls;	      /* Rho^2, <<7	*/
	u32   minrtt_us;      /* Minimum smoothed round trip time value seen */
};

/* BBR congestion control block */
struct bbr {
	u32	min_rtt_us;	        /* min RTT in min_rtt_win_sec window */
	u32	min_rtt_stamp;	        /* timestamp of min_rtt_us */
	u32	probe_rtt_done_stamp;   /* end time for BBR_PROBE_RTT mode */
	struct minmax bw;	/* Max recent delivery rate in pkts/uS << 24 */
	u32	rtt_cnt;	    /* count of packet-timed rounds elapsed */
	u32     next_rtt_delivered; /* scb->tx.delivered at end of round */
	u64	cycle_mstamp;	     /* time of this cycle phase start */
	u32     mode:3,		     /* current bbr_mode in state machine */
		prev_ca_state:3,     /* CA state on previous ACK */
		packet_conservation:1,  /* use packet conservation? */
		round_start:1,	     /* start of packet-timed tx->ack round? */
		idle_restart:1,	     /* restarting after idle? */
		probe_rtt_round_done:1,  /* a BBR_PROBE_RTT round at 4 pkts? */
		unused:13,
		lt_is_sampling:1,    /* taking long-term ("LT") samples now? */
		lt_rtt_cnt:7,	     /* round trips in long-term interval */
		lt_use_bw:1;	     /* use lt_bw as our bw estimate? */
	u32	lt_bw;		     /* LT est delivery rate in pkts/uS << 24 */
	u32	lt_last_delivered;   /* LT intvl start: tp->delivered */
	u32	lt_last_stamp;	     /* LT intvl start: tp->delivered_mstamp */
	u32	lt_last_lost;	     /* LT intvl start: tp->lost */
	u32	pacing_gain:10,	/* current gain for setting pacing rate */
		cwnd_gain:10,	/* current gain for setting cwnd */
		full_bw_reached:1,   /* reached full bw in Startup? */
		full_bw_cnt:2,	/* number of rounds without large bw gains */
		cycle_idx:3,	/* current index in pacing_gain cycle array */
		has_seen_rtt:1, /* have we seen an RTT sample yet? */
		unused_b:5;
	u32	prior_cwnd;	/* prior cwnd upon entering loss recovery */
	u32	full_bw;	/* recent bw, to estimate if pipe is full */

	/* For tracking ACK aggregation: */
	u64	ack_epoch_mstamp;	/* start of ACK sampling epoch */
	u16	extra_acked[2];		/* max excess data ACKed in epoch */
	u32	ack_epoch_acked:20,	/* packets (S)ACKed in sampling epoch */
		extra_acked_win_rtts:5,	/* age of extra_acked, in round trips */
		extra_acked_win_idx:1,	/* current index in extra_acked array */
		unused_c:6;
};

struct westwood {
	u32    bw_ns_est;        /* first bandwidth estimation..not too smoothed 8) */
	u32    bw_est;           /* bandwidth estimate */
	u32    rtt_win_sx;       /* here starts a new evaluation... */
	u32    bk;
	u32    snd_una;          /* used for evaluating the number of acked bytes */
	u32    cumul_ack;
	u32    accounted;
	u32    rtt;
	u32    rtt_min;          /* minimum observed RTT */
	u8     first_ack;        /* flag which infers that this is the first ack */
	u8     reset_rtt_min;    /* Reset RTT min to next RTT sample*/
};

struct veno {
	u8 doing_veno_now;	/* if true, do veno for this rtt */
	u16 cntrtt;		/* # of rtts measured within last rtt */
	u32 minrtt;		/* min of rtts measured within last rtt (in usec) */
	u32 basertt;		/* the min of all Veno rtt measurements seen (in usec) */
	u32 inc;		/* decide whether to increase cwnd */
	u32 diff;		/* calculate the diff rate */
	
};

/* Vegas variables */
struct vegas {
	u32	beg_snd_nxt;	/* right edge during last RTT */
	u32	beg_snd_una;	/* left edge  during last RTT */
	u32	beg_snd_cwnd;	/* saves the size of the cwnd */
	u8	doing_vegas_now;/* if true, do vegas for this RTT */
	u16	cntRTT;		/* # of RTTs measured within last RTT */
	u32	minRTT;		/* min of RTTs measured within last RTT (in usec) */
	u32	baseRTT;	/* the min of all Vegas RTT measurements seen (in usec) */
	u32	cwnd;
};

/* Yeah variables */
/* YeAH variables */
struct yeah {
	
	struct vegas vegas;	/* must be first */

	/* YeAH */
	u32 lastQ;
	u32 doing_reno_now;

	u32 reno_count;
	u32 fast_count;

	u32 pkts_acked;
};

/* CDG variables */
struct cdg_minmax {
	union {
		struct {
			s32 min;
			s32 max;
		};
		u64 v64;
	};
};

struct cdg {
	struct cdg_minmax rtt;
	struct cdg_minmax rtt_prev;
	struct cdg_minmax *gradients;
	struct cdg_minmax gsum;
	bool gfilled;
	u8  tail;
	u8  state;
	u8  delack;
	u32 rtt_seq;
	u32 shadow_wnd;
	u16 backoff_cnt;
	u16 sample_cnt;
	s32 delay_min;
	u32 last_ack;
	u32 round_start;
	
};

/* BIC TCP variables */
struct bic {
	u32	cnt;		/* increase cwnd by 1 after ACKs */
	u32	last_max_cwnd;	/* last maximum snd_cwnd */
	u32	last_cwnd;	/* the last snd_cwnd */
	u32	last_time;	/* time when updated last_cwnd */
	u32	epoch_start;	/* beginning of an epoch */
#define ACK_RATIO_SHIFT	4
	u32	delayed_ack;	/* estimate the ratio of Packets/ACKs << 4 */
	u32	cwnd;
};

/* HTCP variables */
struct htcp {
	u32	alpha;		/* Fixed point arith, << 7 */
	u8	beta;           /* Fixed point arith, << 7 */
	u8	modeswitch;	/* Delay modeswitch
				   until we had at least one congestion event */
	u16	pkts_acked;
	u32	packetcount;
	u32	minRTT;
	u32	maxRTT;
	u32	last_cong;	/* Time since last congestion event end */
	u32	undo_last_cong;

	u32	undo_maxRTT;
	u32	undo_old_maxB;

	/* Bandwidth estimation */
	u32	minB;
	u32	maxB;
	u32	old_maxB;
	u32	Bi;
	u32	lasttime;
};

/* High Speed TCP variables */
struct hstcp {
	u32	ai;
};

/* Illinois TCP variables */
struct illinois {
	u64	sum_rtt;	/* sum of rtt's measured within last rtt */
	u16	cnt_rtt;	/* # of rtts measured within last rtt */
	u32	base_rtt;	/* min of all rtt in usec */
	u32	max_rtt;	/* max of all rtt in usec */
	u32	end_seq;	/* right edge of current RTT */
	u32	alpha;		/* Additive increase */
	u32	beta;		/* Muliplicative decrease */
	u16	acked;		/* # packets acked by current ACK */
	u8	rtt_above;	/* average rtt has gone above threshold */
	u8	rtt_low;	/* # of rtts measurements below threshold */
};

enum PCC_DECISION {
	PCC_RATE_UP,
	PCC_RATE_DOWN,
	PCC_RATE_STAY,
};

enum PCC_MODE {
	PCC_SLOW_START, 
	PCC_DECISION_MAKING,
	PCC_RATE_ADJUSMENT,
	PCC_LOSS, /* When tcp is in loss state, its stats can't be trusted */
};

/* Contains the statistics from one "experiment" interval */
struct pcc_interval {
	u64 rate;

	u32 segs_sent_start;
	u32 segs_sent_end;

	s64 utility;
	u32 lost;
	u32 delivered;
};


struct pcc_data {
	struct pcc_interval *intervals;
	struct pcc_interval *single_interval;
	int send_index;
	int recive_index;

	enum PCC_MODE mode;
	u64 rate;
	u64 last_rate;
	u32 epsilon;
	bool wait_mode;

	enum PCC_DECISION last_decision;
	u32 lost_base;
	u32 delivered_base;

	// debug helpers
	int id;
	int intervals_count;

	u32 segs_sent;
	u32 packets_counted;
	u32 double_counted;

};


// Netlink comm APIs
static void send_msg(char *message, int socketId);
static void start_connection(struct nlmsghdr *nlh);
static void end_connection(struct nlmsghdr *nlh);
static void receive_msg(struct sk_buff *skb);
static int netlink_init(void);
static void netlink_exit(void);

// Mutant state initialization and management
static void save_state(struct sock *sk);
static void load_state(struct sock *sk);
static void init_saved_states(void);
static void print_bictcp(struct bictcp *cubic);
static void print_hybla(struct hybla *hybla);
static void print_bbr(struct bbr *bbr);
static void print_mutant_state(struct sock *sk);

// Methods for TCP congestion control
static void mutant_switch_congestion_control(void);
static void send_net_params(struct tcp_sock *tp, struct sock *sk, int socketId);
static void mutant_tcp_init(struct sock *sk);
static void mutant_tcp_cong_avoid(struct sock *sk, u32 ack, u32 acked);
static u32 mutant_tcp_ssthresh(struct sock *sk);
static void mutant_tcp_set_state(struct sock *sk, u8 new_state);
static u32 mutant_tcp_undo_cwnd(struct sock *sk);
static void mutant_tcp_cwnd_event(struct sock *sk, enum tcp_ca_event event);
static void mutant_tcp_pkts_acked(struct sock *sk, const struct ack_sample *sample);
static u32 mutant_tcp_cong_control(struct sock *sk, const struct rate_sample *rs, u32 ack, u32 acked, int flag);
static u32 mutant_tcp_sndbuf_expand(struct sock *sk);
static u32 mutant_tcp_min_tso_segs(struct sock *sk);
static size_t mutant_tcp_get_info(struct sock *sk, u32 ext, int *attr, union tcp_cc_info *info);
static void mutant_tcp_release(struct sock *sk);
static void send_info(struct mutant_info *info);

#endif /* MUTANT_H */