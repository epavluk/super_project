from functools import reduce
from requests import post
from threading import Thread
from flask import Flask, send_file, send_from_directory
from vidgear.gears import CamGear
from vidgear.gears import WriteGear
from cv2 import applyColorMap, cvtColor, putText, COLORMAP_INFERNO, COLOR_RGB2RGBA, COLOR_RGBA2RGB, FONT_HERSHEY_PLAIN, LINE_AA
from numpy import float32, uint8
from jetson.inference import detectNet
from jetson.utils import cudaFromNumpy, cudaToNumpy

app = Flask(__name__)
net = detectNet("ssd-mobilenet-v2", threshold=0.5)


def frames(hwaccel=False):
    stream = CamGear(source=0).start()
    hwmode = {
        '-vcodec': 'h264_vaapi',
        '-vaapi_device': '/dev/dri/renderD128',
        '-vf': 'format=nv12,hwupload',
    }
    swmode = {
        '-vcodec': 'h264'
    }
    writer_500k = WriteGear(output_filename='500k.m3u8', **{
        **(hwmode if hwaccel else swmode),
        '-b:v': '500k',  # bitrate
        '-pix_fmt': 'yuv420p',
        '-hls_flags': 'omit_endlist+delete_segments',  # live stream
        '-g': '1',
        '-hls_time': '1',
        '-hls_segment_filename': '500k_%d.ts'
    })

    old_count = 0
    while True:
        frame = stream.read()
        if frame is None:
            break

        input_image = cvtColor(frame, COLOR_RGB2RGBA).astype(float32)
        input_image = cudaFromNumpy(input_image)
        detections = net.Detect(input_image, frame.shape[1], frame.shape[0])

        count = 0
        for detection in detections:
            if detection.ClassID == 1:
                count += 1

        if count != old_count:
            print(f'127.0.0.1/api/{count}')
            old_count = count

        frame = cvtColor(cudaToNumpy(
            input_image).astype(uint8), COLOR_RGBA2RGB)
        frame = putText(
            frame, f'peoples: {count}', (0, 40), FONT_HERSHEY_PLAIN, 1, (0, 0, 0))

        writer_500k.write(frame)


@app.after_request
def no_cache(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


@app.route('/')
def index():
    return send_file('index.html')


@app.route('/<path>')
def send(path):
    return send_file(f'{path}')


Thread(target=frames, daemon=True, kwargs={'hwaccel': False}).start()
app.run(host='0.0.0.0', port=8080)
