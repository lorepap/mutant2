import os
import subprocess
import sys


def print_cmd(cmd):
    if isinstance(cmd, list):
        cmd_to_print = ' '.join(cmd).strip()
    elif isinstance(cmd, str):
        cmd_to_print = cmd.strip()
    else:
        cmd_to_print = ''

    if cmd_to_print:
        sys.stderr.write('\n$ %s\n\n' % cmd_to_print)


def print_output(process):

    while True:
        output = process.stdout.readline()

        if output == '' and process.poll() is not None:
            break

        if output:
            print(output.strip())

    rc = process.poll()
    return rc


def call(cmd, **kwargs):
    print_cmd(cmd)
    return subprocess.call(cmd, **kwargs)


def check_call(cmd, **kwargs):
    print_cmd(cmd)
    return subprocess.check_call(cmd, stdout=sys.stdout, **kwargs)


def check_output(cmd, **kwargs):
    print_cmd(cmd)
    return subprocess.check_output(cmd, **kwargs)


def Popen(cmd, **kwargs):
    print_cmd(cmd)
    return subprocess.Popen(cmd, preexec_fn=os.setsid, stdin=subprocess.PIPE, stdout=subprocess.PIPE, **kwargs)