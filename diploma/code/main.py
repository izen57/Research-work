import csv
import random
import tkinter
from tkinter import filedialog, messagebox

import cv2 as cv
import numpy as np
from scipy.spatial import ConvexHull
from sklearn.cluster import AgglomerativeClustering, KMeans

lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
feature_params = dict(maxCorners=0, qualityLevel=0.1, minDistance=3, blockSize=5)
path_to_file: str = ''
colored_features_img = None
colored_centroids_img = None
map_img = None


def build_map() -> None:
    if path_to_file != '':
        App(path_to_file).run()
        global save_colored_clusters_button, watch_colored_clusters_button, save_colored_centroids_button, watch_colored_centroids_button, save_map_button, watch_map_button
        save_colored_clusters_button['state'] = 'normal'
        watch_colored_clusters_button['state'] = 'normal'
        save_colored_centroids_button['state'] = 'normal'
        watch_colored_centroids_button['state'] = 'normal'
        save_map_button['state'] = 'normal'
        watch_map_button['state'] = 'normal'


def open_file() -> None:
    global path_to_file
    path_to_file = filedialog.askopenfilename(
        initialdir='./',
        title='Выберите файл',
        filetypes=[('Video files', '*.avi *.mp4')]
    )
    if path_to_file != '':
        # messagebox.showinfo('Действие выполнено', 'Путь к выбранному файлу: '+path_to_file)
        global build_source_image_button, watch_source_image_button, kmeans_clusters_input, aggl_clusters_input
        build_source_image_button['state'] = 'normal'
        watch_source_image_button['state'] = 'normal'
        kmeans_clusters_input['state'] = 'normal'
        aggl_clusters_input['state'] = 'normal'


def watch_video() -> None:
    if path_to_file == '':
        messagebox.showerror('Невозможно прочесть видеофайл', 'Видеофайл не выбран')
    else:
        cam = cv.VideoCapture(path_to_file)
        while True:
            _, frame = cam.read()
            if frame is None:
                cv.destroyWindow('Video')
                break
            vis = frame.copy()
            cv.imshow('Video', vis)
            ch = cv.waitKey(1)
            if ch == 27:
               break


def watch_image(which: str) -> None:
    if which == 'col_clu':
        cv.imshow('Colored clusters', colored_features_img)
    elif which == 'col_cen':
        cv.imshow('Colored centroids', colored_centroids_img)
    elif which == 'map':
        cv.imshow('Map', map_img)
    _ = cv.waitKey(0)


def save_image(which: str) -> None:
    if which == 'col_clu':
        file_name: str = filedialog.asksaveasfilename(
            initialdir='./',
            initialfile='colored clusters.jpg',
            defaultextension='.jpg',
            title='Сохранить файл',
            filetypes=[('jpg files', '*.jpg')]
        )
        if file_name != '':
            colored_clusters = open(file_name, 'w')
            colored_clusters.close()
            cv.imwrite(file_name, colored_features_img)
    elif which == 'col_cen':
        file_name: str = filedialog.asksaveasfilename(
            initialdir='./',
            initialfile='colored centroids.jpg',
            defaultextension='.jpg',
            title='Сохранить файл',
            filetypes=[('jpg files', '*.jpg')]
        )
        if file_name != '':
            colored_centroids = open(file_name, 'w')
            colored_centroids.close()
            cv.imwrite(file_name, colored_centroids_img)
    elif which == 'map':
        file_name: str = filedialog.asksaveasfilename(
            initialdir='./',
            initialfile='map.jpg',
            defaultextension='.jpg',
            title='Сохранить файл',
            filetypes=[('jpg files', '*.jpg')]
        )
        if file_name != '':
            map = open(file_name, 'w')
            map.close()
            cv.imwrite(file_name, map_img)


def draw_str(dst, target, s) -> None:
    x, y = target
    cv.putText(dst, s, (x+1, y+1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)


def draw_map(file, example_frame) -> None:
    global colored_features_img, colored_centroids_img, map_img
    global kmeans_clusters_input, aggl_clusters_input

    number_of_kmeans_clusters: int = 0
    number_of_aggl_clusters: int = 0
    try:
        number_of_kmeans_clusters = int(kmeans_clusters_input.get())
        number_of_aggl_clusters = int(aggl_clusters_input.get())
        if number_of_aggl_clusters == 0 or number_of_kmeans_clusters < 2:
            raise
    except:
        messagebox.showerror('Ошибка ввода', 'Значение параметра должно быть натуральным числом.')
        return

    # Чтение координат точек интереса из файла в список
    file.seek(0)
    reader = csv.reader(file)
    feature_array = []
    for line in reader:
        x: int = 0
        y: int = 0
        try:
            x = int(line[0])
            y = int(line[1])
        except:
            continue
        feature_array.append([x, y])

    # Определение кластеров точек интереса
    kmeans = KMeans(number_of_kmeans_clusters, n_init='auto', max_iter=300)
    kmeans.fit(feature_array)

    # Раскраска кластеров точек интереса разными цветами
    dict_cluster_color = {}
    for i in range(number_of_kmeans_clusters):
        dict_cluster_color[i] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    colored_features_img = np.zeros_like(example_frame)
    for i, cluster_number in enumerate(kmeans.labels_):
        cv.circle(colored_features_img, (feature_array[i][0], feature_array[i][1]), 0, dict_cluster_color.get(cluster_number), -1)

    # Нахождение центра масс каждого кластера для устранения эффекта тряски камеры
    centroids = kmeans.cluster_centers_
    centroids_hierarchy = AgglomerativeClustering(number_of_aggl_clusters, linkage='average')
    centroids_hierarchy.fit(centroids)
    print(centroids_hierarchy.labels_)

    # Раскраска кластеров центроидов разными цветами
    colored_centroids_img = np.zeros_like(example_frame)
    dict_cluster_color = {}
    for i in range(number_of_aggl_clusters):
        dict_cluster_color[i] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    for i, cluster_number in enumerate(centroids_hierarchy.labels_):
        cv.circle(
            colored_centroids_img,
            (int(centroids[i][0]), int(centroids[i][1])),
            2,
            dict_cluster_color.get(cluster_number),
            -1
        )
        # draw_str(colored_centroids_img, (int(centroids[i][0])+5, int(centroids[i][1])+5), f'{int(centroids[i][0])};{int(centroids[i][1])}')

    # Отрисовка карты-схемы (нахождение выпуклых контуров кластеров точек-центроидов)
    dict_cluster_points = {}
    for i, cluster_number in enumerate(centroids_hierarchy.labels_):
        if cluster_number not in dict_cluster_points:
            dict_cluster_points[cluster_number] = [centroids[i]]
        else:
            dict_cluster_points[cluster_number].append(centroids[i])

    map_img = np.zeros_like(example_frame)
    dict_cluster_hull = {}
    for cluster_number, points in dict_cluster_points.items():
        if len(points) > 2:
            dict_cluster_hull[cluster_number] = ConvexHull(points, True)

    for cluster_number, hull in dict_cluster_hull.items():
        for simplex in hull.simplices:
            x1 = int(hull.points[simplex[0]][0])
            y1 = int(hull.points[simplex[0]][1])
            x2 = int(hull.points[simplex[1]][0])
            y2 = int(hull.points[simplex[1]][1])
            cv.line(
                map_img,
                (x1, y1),
                (x2, y2),
                dict_cluster_color.get(cluster_number)
            )
            # draw_str(map_img, (x1+5, y1+5), f'{x1};{y1}')
            # draw_str(map_img, (x2+5, y2+5), f'{x2};{y2}')


class App:
    def __init__(self, video_src):
        self.track_len = 5
        self.detect_interval = 5
        self.tracks = []
        self.cam = cv.VideoCapture(video_src)
        self.frame_idx = 0

    def run(self):
        file = open('features coordinates.csv', 'w+', newline='')
        writer = csv.writer(file)
        writer.writerow('xy')

        example_frame = None # кадр для инициализации карты по его размерам

        while True:
            _, frame = self.cam.read()
            if frame is None:
                break
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            vis = frame.copy()
            if example_frame is None:
                example_frame = frame.copy()

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, _, _ = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, _, _ = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)

                    cv.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
                    writer.writerow((int(x), int(y)))
                self.tracks = new_tracks
                cv.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                # draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv.circle(mask, (x, y), 5, 0, -1)
                p = cv.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])

            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv.imshow('Optical flow', vis)
            # optical_flow = open('./optical flow.mp4', 'w')
            # optical_flow.close()
            # cv.imwrite('./optical flow.mp4', vis)
            # cv.writeOpticalFlow('./optical flow.mp4', vis)

            ch = cv.waitKey(1)
            if ch == 27:
               break
        cv.destroyWindow('Optical flow')
        draw_map(file, example_frame)
        file.close()


root = tkinter.Tk()
root.title('Построение карты-схемы помещения')
root.resizable(False, False)

root.columnconfigure(index=0, weight=1)
root.columnconfigure(index=1, weight=1)
root.columnconfigure(index=2, weight=1)
root.columnconfigure(index=3, weight=1)
root.columnconfigure(index=4, weight=1)

root.rowconfigure(index=0, weight=1)
root.rowconfigure(index=1, weight=1)
root.rowconfigure(index=2, weight=1)

choose_source_image_button = tkinter.Button(borderwidth=1, relief='solid', text='Выбрать видео', command=open_file)
choose_source_image_button.grid(row=0, column=0, ipadx=6, ipady=6, padx=4, pady=4, sticky='ew')

watch_source_image_button = tkinter.Button(borderwidth=1, relief='solid', text='Посмотреть выбранное видео', command=watch_video)
watch_source_image_button.grid(row=1, column=0, ipadx=6, ipady=6, padx=4, pady=1, sticky='ew')
watch_source_image_button['state'] = 'disable'

build_source_image_button = tkinter.Button(borderwidth=1, relief='solid', text='Построить карту-схему', command=build_map)
build_source_image_button.grid(row=2, column=0, ipadx=6, ipady=6, padx=4, pady=4, sticky='ew')
build_source_image_button['state'] = 'disable'


parameters_labelframe = tkinter.LabelFrame(borderwidth=1, relief='solid', text='Параметры')
parameters_labelframe.grid(row=0, column=1, rowspan=3, ipadx=6, ipady=6, padx=4, pady=4, sticky='ew')

kmeans_clusters_label = tkinter.Label(parameters_labelframe, borderwidth=1, text='Кол-во кластеров для алгоритма k-средних')
kmeans_clusters_label.grid(row=0, column=0, sticky='ew')

kmeans_clusters_input = tkinter.Entry(parameters_labelframe, borderwidth=1, relief='solid')
kmeans_clusters_input.grid(row=1, column=0, ipadx=6, ipady=6, padx=4, pady=4, sticky='ew')
kmeans_clusters_input.insert(0, '200')
kmeans_clusters_input['state'] = 'disable'

aggl_clusters_label = tkinter.Label(parameters_labelframe, borderwidth=1, text='Кол-во кластеров для иерархического алгоритма')
aggl_clusters_label.grid(row=2, column=0, sticky='ew')

aggl_clusters_input = tkinter.Entry(parameters_labelframe, borderwidth=1, relief='solid')
aggl_clusters_input.grid(row=3, column=0, ipadx=6, ipady=6, padx=4, pady=4, sticky='ew')
aggl_clusters_input.insert(0, '1')
aggl_clusters_input['state'] = 'disable'


colored_clusters_labelframe = tkinter.LabelFrame(borderwidth=1, relief='solid', text='Распределение точек по кластерам')
colored_clusters_labelframe.grid(row=0, column=2, rowspan=3, ipadx=6, ipady=6, padx=4, pady=4, sticky='ew')

watch_colored_clusters_button = tkinter.Button(colored_clusters_labelframe, borderwidth=1, relief='solid', text='Посмотреть', command=lambda which='col_clu': watch_image(which))
watch_colored_clusters_button.grid(row=0, column=0, ipadx=6, ipady=6, padx=4, pady=4, sticky='ew')
watch_colored_clusters_button['state'] = 'disable'

save_colored_clusters_button = tkinter.Button(colored_clusters_labelframe, borderwidth=1, relief='solid', text='Сохранить', command=lambda which='col_clu': save_image(which))
save_colored_clusters_button.grid(row=1, column=0, ipadx=6, ipady=6, padx=4, pady=4, sticky='ew')
save_colored_clusters_button['state'] = 'disable'


colored_centroids_labelframe = tkinter.LabelFrame(borderwidth=1, relief='solid', text='Распределение центроидов по кластерам')
colored_centroids_labelframe.grid(row=0, column=3, rowspan=3, ipadx=6, ipady=6, padx=4, pady=4, sticky='ew')

watch_colored_centroids_button = tkinter.Button(colored_centroids_labelframe, borderwidth=1, relief='solid', text='Посмотреть', command=lambda which='col_cen': watch_image(which))
watch_colored_centroids_button.grid(row=0, column=0, ipadx=6, ipady=6, padx=4, pady=4, sticky='ew')
watch_colored_centroids_button['state'] = 'disable'

save_colored_centroids_button = tkinter.Button(colored_centroids_labelframe, borderwidth=1, relief='solid', text='Сохранить', command=lambda which='col_cen': save_image(which))
save_colored_centroids_button.grid(row=1, column=0, ipadx=6, ipady=6, padx=4, pady=4, sticky='ew')
save_colored_centroids_button['state'] = 'disable'


map_labelframe = tkinter.LabelFrame(borderwidth=1, relief='solid', text='Карта-схема')
map_labelframe.grid(row=0, column=4, rowspan=3, ipadx=6, ipady=6, padx=4, pady=4, sticky='ew')

watch_map_button = tkinter.Button(map_labelframe, borderwidth=1, relief='solid', text='Посмотреть', command=lambda which='map': watch_image(which))
watch_map_button.grid(row=0, column=0, ipadx=6, ipady=6, padx=4, pady=4, sticky='ew')
watch_map_button['state'] = 'disable'

save_map_button = tkinter.Button(map_labelframe, borderwidth=1, relief='solid', text='Сохранить', command=lambda which='map': save_image(which))
save_map_button.grid(row=1, column=0, ipadx=6, ipady=6, padx=4, pady=4, sticky='ew')
save_map_button['state'] = 'disable'


root.mainloop()
