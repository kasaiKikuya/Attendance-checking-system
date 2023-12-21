#3a6589eb1f571b17fcfe226eb13256042f9fa629
from flask import Flask, request, jsonify, Response
from PIL import Image
import io
import numpy
import cv2
import socket
import base64
app = Flask(__name__)

@app.route('/api', methods=['POST'])
def message():
    content = request.json
    try:
        payload = request.form.to_dict(flat=False)
        #convert base 64 to image by openCV
        nameimg=0
        im_b64s = payload["images"]
        for im_b64 in im_b64s:
            im_bytes = base64.b64decode(im_b64)
            im_arr = numpy.frombuffer(im_bytes, dtype=numpy.uint8) 
            img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
            #save image 
            nameimg=nameimg+1 
            saveimg=cv2.imwrite('dataFromClient/'+str(nameimg)+'.jpg',img)
        print('gui api anh')
    except Exception as e:
        print(e)
    try:
        command = content['dataFromServer'][0]
        name = content['dataFromServer'][1]
        f = open("exampleRas.txt", "w")
        f.write(command)
        f.write('\n')
        f.write(name)
        f.close()
        print('gui api xac nhan')
    except:
        pass
    try:
        command2 = content['lenhdiemdanh'][0]
        name2 = content['lenhdiemdanh'][1]
        time2 = content['lenhdiemdanh'][2]
        f2 = open("example.txt", "w")
        f2.write(command2)
        f2.write("\n")
        f2.write(name2)
        f2.write("\n")
        f2.write(time2)
        f2.close()
        print('gui api lenh')
    except:
        pass
    return "OK"
if __name__ == "__main__":
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    app.run(host='192.168.0.103') 

    
#bug : sudo lsof -i:5000
#kill -9 56010
