import socket
import time
import datetime
import threading  # thread module changed to threading
import os
import decode_data
import numpy as np

PORT = 5000
SensorIp = '192.168.31.22'
ServerIp = '192.168.31.3'
FILEPATH = './'

T_interval = 1
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = (ServerIp, PORT)
sensor_address = (SensorIp, PORT)

server_socket.bind(server_address)

def all_receive_data(data_time):
    global PORT
    global ServerIp
    global server_socket

    time_tag = get_time_tag()
    old_min = time_tag[14:16]
    file_date = time_tag[0:14]
    writ_data = ''

    old_sec = int(time_tag[17:19]) + T_interval
    old_sec = int(old_sec % 60)

    for ii in range(int(data_time * 50 / 3)):  # added int for Python 3 division
        receive_data, client_address = server_socket.recvfrom(2048)
        time_tag = get_time_tag()

        new_min = time_tag[14:16]
        new_sec = int(time_tag[17:19]) + T_interval
        new_sec = int(new_sec % 60)

        if len(receive_data) < 100:
            print(receive_data)
            ii -= 1
            continue

        if new_sec != old_sec:
            writ_data += time_tag + decode_data.decode_data(receive_data) + '\n'
        else:
            old_sec -= T_interval
            old_min = new_min

            if old_sec < 0:
                old_sec += 60

            filename = ''
            if old_sec < 10:
                filename = file_date + old_min + '_0' + str(old_sec) + '.txt'
            else:
                filename = file_date + old_min + '_' + str(old_sec) + '.txt'

            old_sec = int((new_sec + T_interval) % 60)
            file_date = time_tag[0:14]
            all_data = writ_data

            if len(writ_data) > 100:
                threading.Thread(target=save_file, args=(filename, all_data)).start()
                writ_data = ''

def receive_data():
    global PORT
    global ServerIp
    global server_socket
    receive_data, client_address = server_socket.recvfrom(1024)
    return receive_data

def send_data(data_str):
    global sensor_address
    global server_socket
    server_socket.sendto(data_str.encode(), sensor_address)

def save_file(filename, data):
    global FILEPATH
    date = filename.split('_')[0]

    if not os.path.exists(FILEPATH + 'data/demo/' + date + '/'):
        os.makedirs(FILEPATH + 'data/demo/' + date)

    complete_filename = FILEPATH + 'data/demo/' + date + '/' + filename
    print(complete_filename)
    with open(complete_filename, 'wb') as datafile:
        datafile.write(data.encode())

def get_time_tag():
    timenow = datetime.datetime.now()
    filename = str(timenow).replace(' ', '_').replace(':', '-')
    return filename

if __name__ == "__main__":

    ZERO = chr(0) + chr(0)
    GAIN = 70
    RATE = 10000
    state = 0
    data_time = 3 * 60 * 60  # seconds

    reset_com = 'r' + ZERO
    test_com = 't' + ZERO

    send_data(reset_com)
    send_data(test_com)
    recv_data = receive_data()
    if recv_data==b'T\x00\x00\x00\x00':  # cmp is no longer used in Python 3
        print('test success')
        state = 1
    else:
        print('test err')
        server_socket.close()

    config_com = 'c' + chr(0) + chr(0) + chr(4) + chr(0) + chr(16) + chr(39) + chr(GAIN) + chr(0)
    send_data(config_com)
    recv_data = receive_data()
    if  recv_data == b'C\x00\x00\x01\x00o':
        print('config success')
        state += 1
    else:
        print(recv_data)
        print('config fail')
        server_socket.close()

    start_com = 's' + ZERO + chr(1) + chr(0) + 't'
    stop_com = 's' + ZERO + chr(1) + chr(0) + 'p'

    if state == 2:
        send_data(start_com)
        recv_data = receive_data()
        if recv_data == b'S\x00\x00\x01\x00t':  # it means it starts to update data
            all_receive_data(data_time)
            time.sleep(1)
            send_data(stop_com)
            server_socket.close()
        else:
            print("cannot start")
            server_socket.close()
