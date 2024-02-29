import socket

def get_ip_address(domain):
    return socket.gethostbyname(domain)

domain = "omowill.com"
ip_address = get_ip_address(domain)
print("IP address of", domain, "is", ip_address)