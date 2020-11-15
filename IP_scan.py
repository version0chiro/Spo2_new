from urllib.request import urlopen
import socket
import json
ports = [80, 20, 21, 22, 23, 25, 111, 135, 137, 138, 139, 445, 443, 445, 548, 631, 993, 995]

def get_value(IP):
    while 1:
        try:
            with urlopen(IP) as serverResponse:
                html = serverResponse.read()
                serverResponse.close()
                break
        except:
            pass
    #print(html)
    sensorValues = html.decode()
    return sensorValues

def get_IP(Identifier):
    laptop_ip_address = socket.gethostbyname(socket.gethostname())
    print("IP Address:" + laptop_ip_address)
    val = -1
    substr = "."
    for ind in range(0, 3):
        val = laptop_ip_address.find(substr, val + 1)
    #print(val)
    laptop_ip_address = laptop_ip_address[0:val+1]
    print(laptop_ip_address)

    def connect(hostname, port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        socket.setdefaulttimeout(1)
        result = sock.connect_ex((hostname, port))
        sock.close()
        return result == 0

    available_devices = []
    for i in range(0, 255):
        res = connect(laptop_ip_address+str(i), 80)
        if res:
            # print("Device found at: ", laptop_ip_address+str(i) + ":"+str(80))
            available_devices.append("http://"+laptop_ip_address+str(i))

    # identifying our device
    # print("Available IP Addresses : {}".format(available_devices))
    ipaddress = ""
    for ip in available_devices:
        try:
            ipToPing = ip
            with urlopen(ipToPing) as response:
                dataReceived = response.read()
                response.close()
                dataReceivedFromClient = dataReceived.decode()
                #print(dataReceivedFromClient)
                substring = dataReceivedFromClient.find("AI CAN")
                if substring == -1:
                    pass
                else:
                    ipaddress = str(ip)
                    # print("Device found at IP Address {}".format(ipaddress))
        except:
            pass

    return ipaddress