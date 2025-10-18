import sys


def progbar(i, n, size=16):
    done = (i * size) // n
    bar = ''
    for i in range(size):
        bar += '█' if i <= done else '░'
    return bar


def stream(message) :
    try:
        sys.stdout.write("\r{%s}" % message)
    except:
        
        message = ''.join(i for i in message if ord(i)<128)
        sys.stdout.write("\r{%s}" % message)