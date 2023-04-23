#!/bin/bash
sudo service docker stop
sudo iptables -t nat -A POSTROUTING -s 172.24.0.0/16 ! -o docker0 -j MASQUERADE
sudo service docker start