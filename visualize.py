import os
import time
import torch
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import models.LSTM as LSTM

def parse_data(file):
    with open(f'{file}', 'r') as f:
        data = f.readlines()
    concat_d = []
    for idx, d in enumerate(data):
        d = np.array(d.strip().split("(")[-1].replace(")", "").split(",")).astype(float)
        concat_d.append(d)
        if idx > 10:
            break
    concat_d = np.concatenate(concat_d)
    return concat_d

def normalize(value, min_value=0, max_value=65535, target_min=-1, target_max=1):
    # 스케일링 공식: ((value - min) / (max - min)) * (target_max - target_min) + target_min
    normalized_value = ((value - min_value) / (max_value - min_value)) * (target_max - target_min) + target_min
    return normalized_value

# 데이터 업데이트 및 그래프 그리기 함수
data = []
BATCH_SIZE = 50
model = LSTM.Stacked_BiLSTM(batch_size=BATCH_SIZE, input_size=50, num_layers=1, hidden_size=128, num_direction=2)
model.eval()
def update_graph(frame):
    size = 750*BATCH_SIZE
    if len(data) > size:
        x = torch.FloatTensor(np.array(data[-size:]).reshape(BATCH_SIZE,750,1))
        with torch.no_grad():
            o = model(x).numpy().reshape(-1)
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.plot(data[-size:], linestyle='-')
        plt.title("Input data")
        plt.ylim([-1,1])
        plt.subplot(2, 1, 2)
        plt.plot(o, linestyle='-')
        plt.title("Inferred data")
        plt.xlabel("Sample")
        plt.ylabel("Value")

# 파일 시스템 변화 감지 핸들러
class NewFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        # 새로운 .txt 파일이 추가되었을 때
        if event.is_directory is False and event.src_path.endswith(".txt"):
            print(f"New file detected: {event.src_path}")
            with open(event.src_path, 'r') as f:
                # 파일에서 데이터를 읽어 리스트에 추가
                new_data = parse_data(event.src_path)# [float(line.strip()) for line in f.readlines()]
                data.extend(normalize(new_data))
                
                print(f"Data updated: {new_data}")

# 관찰 시작
def start_watching(path):
    event_handler = NewFileHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# 애니메이션 설정
def start_animation():
    fig = plt.figure(figsize=(18, 6))
    ani = animation.FuncAnimation(fig, update_graph, interval=100)
    plt.show()

# 메인 실행
if __name__ == "__main__":
    folder_to_watch = "./data/demo/2024-09-13/"
    # 실시간 감지와 애니메이션 병렬 실행
    import threading
    watch_thread = threading.Thread(target=start_watching, args=(folder_to_watch,))
    watch_thread.daemon = True
    watch_thread.start()

    # 애니메이션 실행
    start_animation()


# import os
# import time
# import numpy as np
# import matplotlib.pyplot as plt
# from watchdog.observers import Observer
# from matplotlib.animation import FuncAnimation
# from watchdog.events import FileSystemEventHandler



# # 데이터 감시를 위한 핸들러 정의
# class DataHandler(FileSystemEventHandler):
#     def __init__(self, folder_path):
#         self.folder_path = folder_path
#         self.data_files = set([f for f in os.listdir(self.folder_path) if f.endswith('.txt')])
#         self.full_data = []

#     def get_new_data(self):
#         all_files = set([f for f in os.listdir(self.folder_path) if f.endswith('.txt')])
        

#         if self.data_files == all_files:
#             print("No data files found.")
#         else:
#             new_files = all_files - self.data_files
#             print(new_files)
#             self.data_files = all_files
#             return new_files

#     def on_created(self, event):
#         # 새로운 파일이 추가될 때
#         if event.src_path.endswith('.txt'):
#             print(f"New file detected: {event.src_path}")
#             return self.get_new_data()

# # 감시할 폴더 경로
# folder_to_watch = './data/demo/2024-09-10/'

# # 감시자 및 핸들러 설정
# event_handler = DataHandler(folder_to_watch)
# observer = Observer()
# observer.schedule(event_handler, folder_to_watch, recursive=False)

# fig, ax = plt.subplots()
# def animate(i, y):
#     y_data = y

#     x.append(x[-1] + 1)
#     y.append(y_data)

#     x = x[-100:]  # 최근 100개만 봄
#     y = y[-100:]

#     ax.clear()
#     ax.plot(x, y)

# ani = FuncAnimation(fig, animate, fargs=(y), blit=True, interval=1000)
# # 감시자 시작
# observer.start()
# try:
#     plt.show()
# except KeyboardInterrupt:
#     observer.stop()

# observer.join()
