# 삼성전자와 현대자동차 주가로 삼성전자 주가 맞히기

# 각각 데이터에서 컬럼 7개 이상 추출(그 중 거래량은 반드시 들어갈 것)
# timesteps와 feature는 알아서 잘라라

# 제공된 데이터 외 추가 데이터 사용금지

# 1. 삼성전자 28일(화) 종가 맞히기 (점수 배점 0.3)
# 2. 삼성전자 29일(수) 아침 시가 맞히기 (점수 배점 0.7)


#마감시간 : 27일 월 23시 59분 59초        /    28일 화 23시 59분 59초
#윤서희 [삼성 1차] 60,350,07원   (np.round 소수 둘째자리까지)
#윤서희 [삼성]
#첨부파일 : keras53_samsung2_ysh_submit.py       데이터 및 가중치 불러오는 로드가 있어야함
#          keras53_samsung4_ysh_submit.py
#가중치 :  _save/samsung/keras53_samsung2_ysh.h5 / hdf5
#         _save/samsung/keras53_samsung4_ysh.h5 / hdf5