# ifneq ($(KERNELRELEASE),)
# obj-m   := mimic.o
# else
# KDIR    := /lib/modules/$(shell uname -r)/build
# ODIR    := $(shell pwd)/build
# PWD     := /home/lorenzo/Desktop/linux-5.4.231
# MY_CFLAGS += -g -DDEBUG
# # ccflags-y += ${EXTRA_CFLAGS}
# ccflags-y += -I/home/lorenzo/Desktop/linux-5.4.231/include
# # ccflags-y += ${MY_CFLAGS}
# # CC += ${MY_CFLAGS}

# all:
# 		$(MAKE) -C $(KDIR) M=$(shell pwd) modules
# debug:
# 		$(MAKE) -C $(KDIR) M=$(shell pwd) modules
# 		EXTRA_CFLAGS="$(MY_CFLAGS)"

# install:
# 		$(MAKE) -C $(KDIR) M=$(shell pwd) modules_install
# %:
# 		$(MAKE) -C $(KDIR) M=$(shell pwd) $@
# clean:
# 	rm -f *.mod* .*.cmd modules.order Module.symvers Mkfile.old dkms.conf *.o *.ko .mimic.*
# endif

ifneq ($(KERNELRELEASE),)
obj-m   := mutant.o
else
KDIR    := /lib/modules/$(shell uname -r)/build 
ODIR    := $(shell pwd)/build
PWD     := $(shell pwd)
MY_CFLAGS += -g -DDEBUG
ccflags-y += -I/home/lorenzo/Desktop/linux-5.4.231/include

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

debug:
	$(MAKE) -C $(KDIR) M=$(PWD) modules EXTRA_CFLAGS="$(MY_CFLAGS)"

install:
	$(MAKE) -C $(KDIR) M=$(PWD) modules_install

%:
	$(MAKE) -C $(KDIR) M=$(PWD) $@

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean

endif

