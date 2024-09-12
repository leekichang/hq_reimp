import socket
import types
import time
import datetime
import thread
import os
import decode_data
 
PORT = 5000
SensorIp = '192.168.31.22'
ServerIp = '192.168.31.3'
FILEPATH = '/home/pi/'


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
#    while True:
    for ii in range(data_time*50/3): # each package have 600 data,sample rate is 10000,so each second it have 50/3 packages
        receive_data, client_address = server_socket.recvfrom(2048)
        time_tag = get_time_tag()
        new_min = time_tag[14:16]
#        print time_tag 
#        print len(receive_data)
	if len(receive_data) < 100:
	    print receive_data
            ii = ii -1
	    continue 
        
        if new_min == old_min:
            	    
	    writ_data =writ_data + time_tag + decode_data.decode_data(receive_data) +'\n'
	
	else:
	    
            filename = file_date + old_min + '.txt'
	    old_min = new_min
	    file_date = time_tag[0:14]
            
            all_data = writ_data
            thread.start_new_thread(save_file,(filename,all_data))
            writ_data =''   
        #break 
        #print receive_data

def receive_data():
    global PORT
    global ServerIp
    global server_socket
    receive_data, client_address = server_socket.recvfrom(1024)
#    print receive_data
#    print type(receive_data)
    return receive_data

def send_data(data_str):
    global sensor_address
    global server_socket
    server_socket.sendto(data_str.encode(),sensor_address)

def save_file(filename,data):
    global FILEPATH
    date = filename.split('_')[0]
    if os.path.exists(FILEPATH + 'data/' + date +'/') == False:
        os.makedirs(FILEPATH + 'data/' + date)
    complete_filename = FILEPATH + 'data/' + date +'/' +filename
    print complete_filename
    datafile = open(complete_filename,'wb')
    
    #decode data
#    real_data = decode_data.decode_data(data)
    datafile.write(data)
    datafile.close()

def get_time_tag():

    timenow = datetime.datetime.now()
    filename = str(timenow)
    filename = filename.replace(' ','_')
    filename = filename.replace(':','-')
    return filename

if __name__ == "__main__":

    ZERO = chr(0)+chr(0)
    GAIN = 70
    RATE = 10000
    state = 0
    data_time = 3*60*60 # seconds
    
    reset_com = 'r' + ZERO
    test_com ='t' + ZERO
    send_data(reset_com)
    send_data(test_com)
    if cmp(receive_data(),'T'):
        print 'test success'
        state = 1
    else:
        print 'test err'
        server_socket.close()
    
    config_com = 'c' + chr(0) +chr(0) + chr(4)+chr(0) +chr(16) + chr(39)+chr(GAIN) +chr(0)
    send_data(config_com)
    if cmp(receive_data(), 'Co'):
        print 'config success'
        state = state + 1
    else:
        print 'config fail'
        server_socket.close()
    
    start_com  = 's' + ZERO + chr(1)+chr(0) + 't'
    stop_com  = 's' + ZERO + chr(1)+chr(0) + 'p'
    if state == 2:
        send_data(start_com)
        if cmp(receive_data(),'St'): # it means it start to update data
            all_receive_data(data_time)
            time.sleep(1)
            send_data(stop_com)
            server_socket.close()
        else:
            print "can not start"
            server_socket.close()

 
