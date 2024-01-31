#include <asm-generic/int-ll64.h>
#include "../mimic.h"

#ifndef __ARM_OWL_H
#define __ARM_OWL_H 1

static __u32 actions[] = {0, -10, -3, -1, 1, 3, 10};

// void owl_init(struct sock *sk, struct arm *am)
// {
   
// }

// static void owl_acked(struct sock *sk, const struct ack_sample *sample, struct arm *ca)
// {
// }

static __u32 owl_cong_avoid(struct sock *sk, __u32 ack, __u32 acked, u32 action_id)
{
    __u32 new_cwnd;
    __u8 action_id;

    struct tcp_sock *tp = tcp_sk(sk);
    struct arm *am = inet_csk_ca(sk);

    // action_id = 6;
    am->cnt = actions[action_id];

    new_cwnd = tp->snd_cwnd + am->cnt;

    if (new_cwnd < 10)
    {
        am->cnt = 0;
    }

    return am->cnt;
}

#endif /* __ARM_OWL_H */