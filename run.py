#!/usr/bin/python3

import sys, os

os.system("i3-msg 'floating enable; resize set 1187 670; move position center;'")
args = " ".join(sys.argv[1:])
os.system(f"./chip8 {args}")
os.system("i3-msg 'floating disable'")
os.system("clear")
