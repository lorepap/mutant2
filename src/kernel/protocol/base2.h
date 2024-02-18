// #include <linux/inet_diag.h>
#include <net/tcp.h>

// A baseline TCP congestion control algorithm to be used as a reference

static void base2_init(struct sock *sk)
{
    struct tcp_sock *tp = tcp_sk(sk);
    tp->snd_cwnd = 500;
}

static void base2_cong_avoid(struct sock *sk, __u32 ack, __u32 acked)
{
    struct tcp_sock *tp = tcp_sk(sk);
    tp->snd_cwnd = 500;
}

static struct tcp_congestion_ops tcp_base2 __read_mostly = {
	.init		= base2_init,
	.ssthresh	= tcp_reno_ssthresh,
	.undo_cwnd	= tcp_reno_undo_cwnd,
	.cong_avoid	= base2_cong_avoid,
	.owner		= THIS_MODULE,
	.name		= "base2",
};
EXPORT_SYMBOL_GPL(tcp_base2);