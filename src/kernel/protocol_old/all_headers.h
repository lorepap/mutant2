#ifndef __ALL_MIMIC_HEADERS__
#define __ALL_MIMIC_HEADERS__ 1

#include <net/tcp.h>
#include "mimic.h"
#include "protocol/cubic.h"
#include "protocol/hybla.h"
// #include "protocol/owl.h"
#include "protocol/bbr.h"
#include "protocol/vegas.h"

#define ARM_COUNT 4

#define INIT_MSG "0:cubic;1:hybla;2:bbr;3:vegas;"



static void doInit(struct sock *sk, struct arm *ca, __u32 protocolId)
{


      bictcp_init(sk, ca, protocolId);

      hybla_init(sk, ca, protocolId);

    //   bbr_init(sk, ca, protocolId);

      tcp_vegas_init(sk, ca, protocolId);

}

static __u32 mimicCongAvoid(struct sock *sk, __u32 ack, __u32 acked, __u32 protocolId, struct arm *arm){

    // All protocols run at the same time updating their parameters
    // @todo: try to execute sequential for testing but a parallel solution should be implemented
    // only one protocol (protocolID) will modify cwnd or socket related parameters (see inside functions)
    tcp_vegas_cong_avoid(sk, ack, acked, arm, protocolId);
    bictcp_cong_avoid(sk, ack, acked, arm, protocolId);
    hybla_cong_avoid(sk, ack, acked, arm, protocolId);

    // @note: BBR has no cong_avoid function

}


// OLD
// static __u32 changeCongestionWindowSize(struct sock *sk, __u32 ack, __u32 acked, __u32 protocolId, struct arm *arm)
// {
//       switch (protocolId)
//       {
//           case 0:
//               return bictcp_cong_avoid(sk, ack, acked, arm);

//           case 1:
//               return hybla_cong_avoid(sk, ack, acked, arm);

//           case 2:
//               return;

//           case 3:
//               return tcp_vegas_cong_avoid(sk, ack, acked, arm);

//           default:
//               return bictcp_cong_avoid(sk, ack, acked, arm);

//      }
// }

static void doPacketsAcked(struct sock *sk, const struct ack_sample *sample, __u32 protocolId, struct arm *arm)
{
    
    // All protocols are executed sequentially (parallel would be best)
    // @todo: overhead?!
    bictcp_acked(sk, sample, arm, protocolId); // it changes ssthresh internally so protocolID is needed
    tcp_vegas_pkts_acked(sk, sample, arm); // doesn't change any parameter (protocol ID not needed)
    
    // OLD
    //   switch (protocolId)
    //   {

    //       case 0:
    //           return bictcp_acked(sk, sample, arm);

    //       case 1:
    //           return;

    //       case 2:
    //           return;

    //       case 3:
    //           return tcp_vegas_pkts_acked(sk, sample, arm);

    //       default:
    //           return bictcp_acked(sk, sample, arm);

    //  }
}

// This is only implemented in bbr (kernel v5.4.0.131)
static void mimicMain(struct sock *sk, const struct rate_sample *rs, __u32 protocolId, struct arm *ca){

    switch (protocolId)
      {

          case 0:
              return;

          case 1:
              return;

          case 2:
              return bbr_main(sk, sample, ca);

          case 3:
              return;

          default:
              return;

     }
}

static __u32 mimicSsthresh(struct sock *sk, __u32 protocolId, struct arm *ca){
    // @note
    // No protocol will modify tcp socket internally (only reading operations). 
    // Each protocol function returns the current ssthresh value and updates internal parameters.
    // Each update will be reflected on the corresponding protocol variable within the global mimic struct (arm) defined in `mimic.h`
    // Switch-case is needed here because only one value of the ssthresh has to be returned according to the protocol ID
    switch (protocolId)
    {
        case 0:
            return bictcp_recalc_ssthresh(sk, ca);

        case 1:
            return tcp_reno_sshthresh(sk);

        case 2:
            return bbr_ssthresh(sk);

        case 3:
            return tcp_reno_sshthresh(sk);

        default:
            return bictcp_recalc_ssthresh(sk);
   }
}

static void mimicUndoCwnd(struct sock *sk, __u32 protocolId, struct arm *ca){
    // Same as `mimicSsThresh` function
    switch (protocolId)
      {
          case 0:
              return tcp_reno_undo_cwnd(sk);

          case 1:
              return tcp_reno_undo_cwnd(sk);

          case 2:
              return bbr_undo_cwnd(sk, ca);

          case 3:
              return tcp_reno_undo_cwnd(sk);

          default:
              return;

     }

}

static void mimicCongestionEvent(struct sock *sk, enum tcp_ca_event event, __u32 protocolId, struct arm *ca){
    // Only reading operation on socket structure
    // All functions must be executed as they're not returning any value.
    // Functions internally modify their internal variables defined in the global mimic structure in `mimic.h`.
    // Only protocols that support a congestion event are executed (e.g., hybla does not have any cong_event function)

    bictcp_cwnd_event(sk, event, ca);
    bbr_cwnd_event(sk, event, ca);
    tcp_vegas_cwnd_event(sk, event, ca);

    // switch (protocolId)
    //   {
    //       case 0:
    //           return bictcp_cwnd_event(sk, event, ca);

    //       case 1:
    //           return;

    //       case 2:
    //           return bbr_cwnd_event(sk, event, ca);

    //       case 3:
    //           return tcp_vegas_cwnd_event(sk, event, ca);

    //       default:
    //           return bictcp_cwnd_event(sk, event, ca);

    //  }
}

static void mimicState(struct sock *sk, __u8 new_state, struct arm *ca){
    // This function is executed on cwnd event (see tcp_congestion_ops struct)

    bictcp_state(sk, new_state, ca);
    bbr_set_state(sk, new_state, ca);
    tcp_vegas_state(sk, new_state, ca)+
    hybla_state(sk, new_state, ca);

    // switch (protocolId)
    //   {
    //       case 0:
    //           return bictcp_cwnd_event(sk, event, ca);

    //       case 1:
    //           return ;

    //       case 2:
    //           return bbr_cwnd_event(sk, event, ca);

    //       case 3:
    //           return;

    //       default:
    //           return bictcp_cwnd_event(sk, event);

    //  }
}

static __u32 mimicSndbufExpand(struct sock *sk){

    bbr_sndbuf_expand(*sk);

}

#endif
