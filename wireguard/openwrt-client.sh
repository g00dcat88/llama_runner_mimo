#!/bin/sh
# WireGuard клиент для OpenWrt → Windows PC сервер

# Очистить старый конфиг
uci delete network.wg0 2>/dev/null
uci commit network

# Интерфейс
uci set network.wg0=interface
uci set network.wg0.proto='wireguard'
uci set network.wg0.private_key='aCK7DNDfJs2NPIcnHMrC0PdseubjQqrRYkQZ8n8ORHg='
uci add_list network.wg0.addresses='10.10.0.2/24'

# Пир (Windows PC сервер)
uci set network.lstart=wireguard_wg0
uci set network.lstart.description='L-start PC'
uci set network.lstart.public_key='dOB97MKJ0ZZMYVkqjnDtzJ22cAJlxrnoCzenhp7It3U='
uci set network.lstart.preshared_key='Lh93CCygMMe2SE0OEf9QXaWJ5Oas8G9Dve/DhEyeux0='
uci set network.lstart.endpoint_host='95.174.126.186'
uci set network.lstart.endpoint_port='51820'
uci set network.lstart.persistent_keepalive='25'
uci add_list network.lstart.allowed_ips='10.10.0.1/32'

uci commit network
/etc/init.d/network restart
