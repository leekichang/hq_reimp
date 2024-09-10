from struct import *
import types
def decode_data(bin_data):
    head_info = bin_data[0:17]  # 17 bytes are head info
    head = unpack('<cHHIIHH',head_info)
    #print head
    data = bin_data[17::]
    #print len(data)
    real_data = unpack('<600H',data)
    #print type(head)
    #print type(real_data)
    return str(head) + str(real_data)


if __name__  == "__main__":
    filename = '2018-01-15_16-30-00.644228.txt'
    f = open(filename,'rb')
    bin_data = f.read()
    print len(bin_data)
    f.close()
    decode_data(bin_data)

