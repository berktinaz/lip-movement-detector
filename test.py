import detector

obj = detector.BlinkDetector("predictor/shape_predictor_68_face_landmarks.dat")

frames, m_ratio, points = obj.process_video('example/test.mp4',
                                                  'csv/test.csv')
