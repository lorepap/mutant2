// #include <linux/inet_diag.h>
#include <net/tcp.h>

// A baseline TCP congestion control algorithm to be used as a reference

static void base_init(struct sock *sk)
{
    struct tcp_sock *tp = tcp_sk(sk);
    tp->snd_cwnd = 10;
}

static void base_cong_avoid(struct sock *sk, __u32 ack, __u32 acked)
{
    struct tcp_sock *tp = tcp_sk(sk);
    tp->snd_cwnd = 10;
}

static struct tcp_congestion_ops tcp_base __read_mostly = {
	.init		= base_init,
	.ssthresh	= tcp_reno_ssthresh,
	.undo_cwnd	= tcp_reno_undo_cwnd,
	.cong_avoid	= base_cong_avoid,
	.owner		= THIS_MODULE,
	.name		= "base",
};
EXPORT_SYMBOL_GPL(tcp_base);