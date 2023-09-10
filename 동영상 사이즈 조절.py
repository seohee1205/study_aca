import cv2

def resize_video(video_path, output_path, width, height):
    # 비디오 파일 열기
    video_capture = cv2.VideoCapture(video_path)

    # 비디오의 속성 가져오기
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    # 비디오 라이터 생성
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정 (여기서는 mp4v)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        # 비디오에서 프레임 읽기
        ret, frame = video_capture.read()

        if not ret:
            break

        # 프레임 크기 변경
        resized_frame = cv2.resize(frame, (width, height))

        # 변경된 프레임을 비디오에 쓰기
        out.write(resized_frame)

    # 사용한 객체들 해제
    video_capture.release()
    out.release()

    print("영상 변환 완료.")

# 사용 예시
input_video = 'C:/_study/team_project/Traffic_light/횡단보도.mp4'  # 입력 비디오 경로
output_video = 'C:/_study/team_project/Traffic_light/횡단보도_변환.mp4'  # 출력 비디오 경로
width = 640  # 변경할 너비
height = 640  # 변경할 높이

resize_video(input_video, output_video, width, height)