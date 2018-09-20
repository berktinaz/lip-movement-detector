from imutils import face_utils
from scipy.spatial import distance as dist
from skimage.color import rgb2grey
from skimage import img_as_ubyte
import csv
import cv2
import dlib
import heapq
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.signal as sci
import skvideo.io


class LipMovementDetector:
    def __init__(self, predictor):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor)
        self.mouth_landmarks = slice(*face_utils.FACIAL_LANDMARKS_IDXS["mouth"])
        self.MOUTH_PEAK_CONST_SPEECH = 80
        self.MOUTH_PEAK_CONST_DIGIT = 30
        self.MOUTH_RUNNING_MEAN_WIDTH = 30
        self.MOUTH_WIDTH = 65
        self.MOUTH_HEIGHT = 45
        self.MOUTH_THRESH = 0
        self.MOUTH_POINT_SEARCH_WIDTH = 10
        self.MOUTH_ANNOTATION_THRESH = 10
        self.PEAK_AMOUNT = 10

    # Static Methods

    @staticmethod
    def clean_peaks(peaks, number):  # take largest N

        if len(peaks[0]) <= number:
            return peaks

        else:
            cleared_peaks = heapq.nlargest(number, peaks[0])
            return cleared_peaks, {k: peaks[1][k] for k in cleared_peaks if k in peaks[1]}

    @staticmethod
    def clean_peaks_alternative(peaks, number):  # take first N

        if len(peaks[0]) <= number:
            return peaks

        else:
            cleared_peaks = peaks[0][0:number]
            return cleared_peaks, {k: peaks[1][k] for k in cleared_peaks if k in peaks[1]}

    @staticmethod
    def clean_peaks_alternative2(peaks, number):  # take N that spans furthest

        if len(peaks[0]) <= number:
            return peaks

        else:
            maxi = 0
            selected_set = []

            subsets = set(itertools.combinations(peaks[0], number))

            for temp_set in subsets:
                distance = sum([temp_set[i] - temp_set[i - 1] for i in range(1, len(temp_set))])
                if distance > maxi:
                    selected_set = temp_set
                    maxi = distance

            return selected_set, {k: peaks[1][k] for k in selected_set if k in peaks[1]}

    @staticmethod
    def clean_peaks_alternative2_revised(peaks, number):  # take N that spans furthest

        if len(peaks[0]) <= number:
            return peaks

        else:
            minimum = 1000000000000  # initialize with high number
            selected_set = []

            subsets = set(itertools.combinations(peaks[0], number))

            for temp_set in subsets:
                distance = sum([(temp_set[i] - temp_set[i - 1]) ** 2 for i in range(1, len(temp_set))])
                if distance < minimum:
                    selected_set = temp_set
                    minimum = distance

            return selected_set, {k: peaks[1][k] for k in selected_set if k in peaks[1]}

    @staticmethod
    def divide_video(points, video):
        name = 0
        for index in range(points[0].shape[0]):
            if index == points[0].shape[0] - 1:
                skvideo.io.vwrite(str(name) + '.mp4', video[points[0][index]:, :, :, :],
                                  outputdict={"-vcodec": "libx264"})
                break
            skvideo.io.vwrite(str(name) + '.mp4', video[points[0][index]: points[0][index + 1], :, :, :],
                              outputdict={"-vcodec": "libx264"})
            name += 1

    @staticmethod
    def draw_points(image, points, tag=True, in_place=False, color=(255, 255, 255)):
        font = cv2.FONT_HERSHEY_SIMPLEX

        if in_place:
            img = image
        else:
            img = np.copy(image)

        for i in range(points.shape[0]):
            if tag:
                cv2.putText(img, str(i), (int(points[i, 0]), int(points[i, 1])), font, 0.23, color)
            else:
                cv2.circle(img, (int(points[i, 0]), int(points[i, 1])), 1, color)
        return img

    @staticmethod
    def get_absolute_error(list1, list2):

        err = 0
        if len(list1) != len(list2):
            return -1

        else:
            if len(list1) == 0:
                return 0

            for i in range(len(list1)):
                err += math.fabs(list1[i] - list2[i])
            return err / len(list1)

    @staticmethod
    def get_best_mean_width(signal, lower_bound, upper_bound, ground_truth, peak_const=1):

        error = 1000000000  # initialize with high number
        width = 0

        for i in range(lower_bound, upper_bound):
            signal_temp = LipMovementDetector.smooth_signal(signal, i)

            points = sci.find_peaks(signal_temp, 0, None, peak_const)

            temp_error = LipMovementDetector.get_meansquare_error(points[0], ground_truth)
            print(temp_error)

            if temp_error != -1 and temp_error < error:
                error = temp_error
                width = i

        return width

    @staticmethod
    def get_derivative(signal):
        return [(signal[i + 1] - signal[i]) for i in range(len(signal) - 1)]

    @staticmethod
    def get_mar(mouth_landmarks):

        # inner points
        vert_dist1 = dist.euclidean(mouth_landmarks[13, :], mouth_landmarks[19, :])
        vert_dist2 = dist.euclidean(mouth_landmarks[14, :], mouth_landmarks[18, :])
        vert_dist3 = dist.euclidean(mouth_landmarks[15, :], mouth_landmarks[17, :])

        hor_dist = dist.euclidean(mouth_landmarks[12, :], mouth_landmarks[16, :])

        mar = (vert_dist1 + vert_dist2 + vert_dist3) / (3.0 * hor_dist)
        return mar

    @staticmethod
    def get_meansquare_error(list1, list2):

        err = 0
        if len(list1) != len(list2):
            return -1
        for i in range(len(list1)):
            err += (list1[i] - list2[i]) ** 2
        return (err / len(list1)) ** 0.5

    @staticmethod
    def get_mouth_area(marks):
        marks = marks[12:, :]
        n = marks.shape[0]  # num of points
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += marks[i, 0] * marks[j, 1]
            area -= marks[j, 0] * marks[i, 1]
        area = abs(area) / 2.0
        return area

    @staticmethod
    def get_mouth_center(mouth_marks):
        return int((mouth_marks[0, 0] + mouth_marks[6, 0]) / 2), int((mouth_marks[3, 1] + mouth_marks[9, 1]) / 2)

    @staticmethod
    def magnitude_sum(mag):

        mag_sum = 0

        for i in range(mag.shape[0]):
            for j in range(mag.shape[1]):
                mag_sum += mag[i, j]

        return mag_sum

    @staticmethod
    def plot_signals(signal, signal_der, detected, frames, ground_truth):
        plt.figure(1)
        plt.subplot(2, 1, 1)

        plt.title("Abs Error: " + str(LipMovementDetector.get_absolute_error( detected[0], ground_truth)))

        plt.scatter(detected[0], [signal[i] for i in detected[0]], color='r')

        plt.scatter(ground_truth, [signal[i] for i in ground_truth], color='g')
        plt.plot(frames, signal, '.-')
        plt.ylabel('Signal')

        plt.subplot(2, 1, 2)
        plt.scatter(detected[0], detected[1].get('peak_heights'), color='r')
        plt.plot(frames[:-1], signal_der, '.-')
        plt.ylabel('Derivative')
        plt.xlabel('Frame')

        plt.show()

    @staticmethod
    def plot_vectors(mag, ang, img):
        plt.gcf().clear()
        plt.ion()
        x, y = np.linspace(0, mag.shape[1] - 1, mag.shape[1]), np.linspace(0, mag.shape[0] - 1, mag.shape[0])
        dx, dy = np.multiply(mag, np.cos(ang)), np.multiply(mag, np.sin(ang))

        plt.quiver(x, y, dx, dy)
        plt.imshow(img)
        plt.show()
        plt.pause(0.3)

    @staticmethod
    def read_annotated_points_from_csv(filename, fps=30):

        with open(filename, newline='') as landmarks_file:
            landmark_reader = csv.reader(landmarks_file, delimiter=',', quotechar='|')

            start_points = []
            end_points = []
            utterances = []
            ignore_first = True

            for row in landmark_reader:
                if ignore_first:
                    ignore_first = False
                    continue
                if row[4] == "g":
                    continue

                start_points.append( max( 0, round( int(row[0]) * fps / (10**7)) - 1) )
                end_points.append( max( 0, round( int(row[1]) * fps / (10**7)) - 1) )
                utterances.append( row[4] )

        return start_points, end_points, utterances

    @staticmethod
    def read_landmarks_from_csv(filename, landmark_type='2d'):

        with open(filename, newline='') as landmarks_file:
            landmark_reader = csv.reader(landmarks_file, delimiter=',', quotechar='|')

            landmarks = []

            for row in landmark_reader:

                frame_landmarks = []

                row = row[1:]

                if landmark_type == '3d':

                    for index in range(0, len(row), 3):

                        coordinates = [int(row[index + 2]), int(row[index + 1])]

                        frame_landmarks.append(coordinates)

                else:

                    for index in range(0, len(row), 2):
                        coordinates = [int(row[index + 1]), int(row[index])]

                        frame_landmarks.append(coordinates)


                landmarks.append(frame_landmarks)

        return np.array(landmarks)

    @staticmethod
    def smooth_signal(x, n):
        return np.convolve(x, np.ones((n,)) / n)[(n - 1):]

    @staticmethod
    def vectoral_sum(mag, ang):
        x = 0
        y = 0

        for i in range(mag.shape[0]):
            for j in range(mag.shape[1]):
                x += math.cos(ang[i, j]) * mag[i, j]
                y += math.sin(ang[i, j]) * mag[i, j]

        return (x ** 2 + y ** 2) ** 0.5

    @staticmethod
    def visualize_points(video_file, csv_file, visualize_with_numbers=False, save_video=False,
                         output_filename='default.mp4', landmark_type='2d'):

        cap = cv2.VideoCapture(video_file)
        all_landmarks = LipMovementDetector.read_landmarks_from_csv(csv_file, landmark_type)
        frame_no = 0

        if save_video:
            out = skvideo.io.FFmpegWriter(output_filename, inputdict={},
                                          outputdict={'-vcodec': 'libx264', '-pix_fmt': 'rgb24', '-r': '30'})

        while cap.isOpened():

            ret, frame = cap.read()

            if ret:

                preds = all_landmarks[frame_no]
                frame_no += 1

                temp_img = LipMovementDetector.draw_points(frame, preds, tag=visualize_with_numbers)

                cv2.imshow('Frame', temp_img)
                cv2.waitKey(29)

                if save_video:
                    out.writeFrame(cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB))

            else:
                break

        cap.release()

        if save_video:
            out.close()

    # Methods
    def cut_mouth_from_img(self, img, landmarks, mouth_height, mouth_width, center_x=None, center_y=None):

        mouth_marks = landmarks[self.mouth_landmarks]

        if center_x is None and center_y is None:
            center_x, center_y = self.get_mouth_center( mouth_marks)

        cutted_img = np.copy(img[ int(round(center_y) - round(mouth_height)): int(round(center_y) + round(mouth_height)),
                             int(round(center_x) - round(mouth_width)): int(round(center_x) + round(mouth_width))])

        return cutted_img

    def cut_mouth_from_video(self, video, all_landmarks, mouth_padding_height, mouth_padding_width, output_filename='default.mp4', mouth_height=None, mouth_width=None):

        if mouth_width and mouth_height is None:

            mouth_height = 0
            mouth_width = 0

            for landmarks in all_landmarks:

                mouth_marks = landmarks[self.mouth_landmarks]
                mouth_width = max( mouth_width, (mouth_marks[6, 0] - mouth_marks[0, 0])/2 )
                mouth_height = max( mouth_height, (mouth_marks[9, 1] - mouth_marks[3, 1])/2 )

        out = skvideo.io.FFmpegWriter(output_filename, inputdict={},
                                      outputdict={'-vcodec': 'libx264', '-pix_fmt': 'rgb24', '-r': '30'})

        for frame_no in range(video.shape[0]):

            cropped_frame = self.cut_mouth_from_img( video[frame_no], all_landmarks[frame_no], mouth_height + mouth_padding_height, mouth_width + mouth_padding_width )
            out.writeFrame(cropped_frame)

        out.close()

    def cut_mouth_from_video_smoother(self, video, all_landmarks, mouth_padding_height, mouth_padding_width, window_size=10, output_filename='default.mp4', mouth_height=None, mouth_width=None):

        unfinished_flag = False

        if mouth_width and mouth_height is None:

            mouth_height = 0
            mouth_width = 0

            for landmarks in all_landmarks:

                mouth_marks = landmarks[self.mouth_landmarks]
                mouth_width = max( mouth_width, (mouth_marks[6, 0] - mouth_marks[0, 0])/2 )
                mouth_height = max( mouth_height, (mouth_marks[9, 1] - mouth_marks[3, 1])/2 )

        out = skvideo.io.FFmpegWriter(output_filename, inputdict={},
                                      outputdict={'-vcodec': 'libx264', '-pix_fmt': 'rgb24', '-r': '30'})

        for frame_no in range(video.shape[0]):

            if frame_no + window_size <= video.shape[0]:

                tot_center_x = 0
                tot_center_y = 0

                for new_frame_no in range(frame_no, frame_no + window_size):

                    landmarks = all_landmarks[new_frame_no]
                    mouth_marks = landmarks[self.mouth_landmarks]

                    new_center_x, new_center_y = self.get_mouth_center( mouth_marks)
                    tot_center_y += new_center_y
                    tot_center_x += new_center_x

                mean_center_y = tot_center_y / window_size
                mean_center_x = tot_center_x / window_size

                cropped_frame = self.cut_mouth_from_img(video[frame_no], all_landmarks[frame_no],
                                                        mouth_height + mouth_padding_height,
                                                        mouth_width + mouth_padding_width, mean_center_x, mean_center_y)

                try:
                    out.writeFrame(cropped_frame)
                except ValueError:
                    unfinished_flag = True
                    out.close()
                    return unfinished_flag

            else:
                cropped_frame = self.cut_mouth_from_img(video[frame_no], all_landmarks[frame_no],
                                                        mouth_height + mouth_padding_height,
                                                        mouth_width + mouth_padding_width, mean_center_x, mean_center_y)
                try:
                    out.writeFrame(cropped_frame)
                except ValueError:
                    unfinished_flag = True
                    out.close()
                    return unfinished_flag

        out.close()
        return unfinished_flag

    def get_mar_and_area(self, landmarks):

        mouth_marks = landmarks[self.mouth_landmarks]

        mar = self.get_mar(mouth_marks)
        m_area = self.get_mouth_area(mouth_marks)

        return mar, m_area

    def get_mouth_optical_flow(self, video_file, ground_truth):

        signal = []
        frames = []
        frame_no = 0

        cap = cv2.VideoCapture(video_file)
        video = skvideo.io.vread(video_file)

        ret, frame1 = cap.read()

        rects = self.detector(frame1, 0)

        landmarks = self.predictor(frame1, rects[0])
        landmarks = face_utils.shape_to_np(landmarks)

        frame1 = self.cut_mouth_from_img(frame1, landmarks, self.MOUTH_HEIGHT, self.MOUTH_WIDTH)

        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        # debug
        print(frame1.shape[0])
        print(frame1.shape[1])

        while cap.isOpened():

            ret, frame2 = cap.read()

            if ret:

                frames.append(frame_no)

                rects = self.detector(frame2, 0)

                landmarks = self.predictor(frame2, rects[0])
                landmarks = face_utils.shape_to_np(landmarks)

                frame2 = self.cut_mouth_from_img(frame2, landmarks, self.MOUTH_HEIGHT, self.MOUTH_WIDTH)

                nextf = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

                flow = cv2.calcOpticalFlowFarneback(prvs, nextf, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                print(mag.shape[0])

                self.plot_vectors(mag, ang, cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))

                signal.append(self.vectoral_sum(mag, ang))

                prvs = nextf
                frame_no += 1

            else:
                break

        cap.release()
        cv2.destroyAllWindows()

        # signal = self.smooth_signal( signal, 5)
        der = np.array(self.get_derivative(signal))  # use derivative

        index = self.get_best_mean_width(der, 1, 50, ground_truth)
        print(index)  # debug

        smoothed_signal = self.smooth_signal(der, index)
        points = sci.find_peaks(smoothed_signal, 0, None, self.MOUTH_PEAK_CONST)

        self.divide_video(points, video)

        return frames, signal, points, index

    def process_video(self, video_file=None, csvfile=None):
        video = skvideo.io.vread(video_file)

        m_ratio = []
        m_area = []
        frames = []

        if csvfile is None:
            for frame_no in range(video.shape[0]):
                grey_frame = img_as_ubyte(rgb2grey(video[frame_no]))

                # detect the face
                rects = self.detector(grey_frame, 0)

                landmarks = self.predictor(grey_frame, rects[0])
                landmarks = face_utils.shape_to_np(landmarks)

                mar, mouth_area = self.get_mar_and_area(landmarks)

                frames.append(frame_no)
                m_ratio.append(mar)
                m_area.append(mouth_area)

        else:
            all_landmarks = self.read_landmarks_from_csv(csvfile)
            for frame_no in range(all_landmarks.shape[0]):
                landmarks = all_landmarks[frame_no]

                mar, mouth_area = self.get_mar_and_area(landmarks)

                frames.append(frame_no)
                m_ratio.append(mar)
                m_area.append(mouth_area)

        der = np.array(self.get_derivative(m_ratio))  # use derivative

        s_signal = self.smooth_signal(der, self.MOUTH_RUNNING_MEAN_WIDTH)
        points = sci.find_peaks(s_signal, self.MOUTH_THRESH, None, self.MOUTH_PEAK_CONST_SPEECH)

        return frames, m_ratio, points

    def refine_annotations(self, landmark_csv, annotation_csv):

        frames = []
        m_ratio = []
        refined_points = []

        all_landmarks = self.read_landmarks_from_csv(landmark_csv)
        for frame_no in range(all_landmarks.shape[0]):
            landmarks = all_landmarks[frame_no]

            mar, mouth_area = self.get_mar_and_area(landmarks)

            frames.append(frame_no)
            m_ratio.append(mar)

        # get annotated end points
        _, annotated_points, utterances = self.read_annotated_points_from_csv( annotation_csv)

        for point_index in range(len(annotated_points)):
            lower_bound = annotated_points[point_index]
            upper_bound = min(len(m_ratio), annotated_points[point_index] + self.MOUTH_POINT_SEARCH_WIDTH)

            cutted_signal = np.negative( m_ratio[lower_bound:upper_bound] ) + max(m_ratio)

            points = sci.find_peaks(cutted_signal, self.MOUTH_THRESH, None, 5)

            if point_index == len(annotated_points) - 1:
                refined_points.append( annotated_points[point_index] )
                break

            first_flag = True

            if len(points[0]) != 0:

                for temp_point in points[0]:
                    if first_flag:
                        refined_point = temp_point
                        first_flag = False
                        continue
                    if abs(temp_point + lower_bound - annotated_points[point_index]) < abs(refined_point + lower_bound - annotated_points[point_index]):
                        refined_point = temp_point

            else:

                refined_point = 0

            if utterances[point_index] == 'g':
                refined_points.remove( refined_points[-1])
                utterances.remove('g')

            refined_points.append(refined_point + lower_bound)

        return refined_points, utterances

    def refine_annotations_from_direc( self, annotation_direc, landmark_direc ):

        for root, dirs, filenames in os.walk(annotation_direc):
            for filename in filenames:
                if (filename[-4:] == '.csv') and (filename[-7] == 'R'):

                    annotation_name = os.path.join(root, filename)
                    landmark_name = landmark_direc + root[len(annotation_direc):] + '/' + filename[:-4] + '_00D.csv'
                    refined_annotation_name = annotation_name[:-4] + ' (refined).csv'

                    refined_points, utterances = self.refine_annotations( landmark_name, annotation_name)

                    with open(refined_annotation_name, 'w', newline='') as csvfile:
                        newcsv = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

                        title_row = ['Relative_Start_Time', 'Relative_Stop_Time', 'Utterance']
                        newcsv.writerow( title_row)

                        for index in range(len(refined_points)):

                            print(int(float("%.3f" % ((refined_points[index] + 1) / 30)) * 1000)*10000)

                            if index == 0:
                                temp_row = [0, int(float("%.3f" % ((refined_points[index] + 1) / 30)) * 1000)*10000, utterances[index]]
                            else:
                                temp_row = [int(float("%.3f" % ((refined_points[index-1] + 1) / 30)) * 1000)*10000, int(float("%.3f" % ((refined_points[index] + 1) / 30)) * 1000)*10000, utterances[index]]

                            newcsv.writerow(temp_row)

    def refine_annotations_from_direc_v2( self, annotation_direc, landmark_direc, output_direc):

        for root, dirs, filenames in os.walk(annotation_direc):

            structure = output_direc + root[len(annotation_direc):]

            if not os.path.isdir(structure):
                os.mkdir(structure)
            else:
                print( structure + " does already exits!")

            for filename in filenames:
                if (filename[-4:] == '.csv') and (filename[-7] == 'R'):

                    print(filename)

                    refined_annotation_name = structure + '/' + filename[:-4] + ' (refined).csv'

                    if not os.path.isfile(refined_annotation_name):

                        annotation_name = os.path.join(root, filename)
                        landmark_name = landmark_direc + root[len(annotation_direc):] + '/' + filename[:-4] + '_00D.csv'

                        refined_points, utterances = self.refine_annotations( landmark_name, annotation_name)

                        with open(refined_annotation_name, 'w', newline='') as csvfile:
                            newcsv = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

                            title_row = ['Relative_Start_Time', 'Relative_Stop_Time', 'Absolute_Start_Time', 'Absolute_Stop_Timem', 'Utterance']
                            newcsv.writerow( title_row)

                            for index in range(len(refined_points)):

                                if index == 0:
                                    temp_row = [0, int(float("%.3f" % ((refined_points[index] + 1) / 30)) * 1000)*10000, 0, 0, utterances[index]]
                                else:
                                    temp_row = [int(float("%.3f" % ((refined_points[index-1] + 1) / 30)) * 1000)*10000, int(float("%.3f" % ((refined_points[index] + 1) / 30)) * 1000)*10000, 0, 0, utterances[index]]

                                newcsv.writerow(temp_row)

                    else:
                        print( refined_annotation_name + "exists")
                        continue
