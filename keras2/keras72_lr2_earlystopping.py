# [실습] earlystopping 적용하려면 어떻게 하면 될까요?
#1. 최소값을 넣을 변수를 하나, 카운트 할 변수 하나 준비
#2. 다음 에포에 값과 최소값을 비교, 최소값이 갱신되면 
#   그 변수에 최소값을 넣어주고, 카운트변수 초기화
#3. 갱신이 안 되면 카운트 변수 ++1
#   카운트 변수가 내가 원하는 얼리스타핑 개수에 도달하면 for문을 stop

x = 10
y = 10
w = 1111
lr = 0.1
epochs = 30000

best_loss = float('inf')  # Initialize with a very large value
patience = 10  # Number of epochs to wait before early stopping
no_improvement = 0  # Counter to keep track of epochs without improvement

for i in range(epochs):
    hypothesis = x * w
    loss = (hypothesis - y) ** 2  # MSE
    
    print('epochs:', i, '\t', 'Loss:', round(loss, 4), '\t', 'Predict:', round(hypothesis, 4))
    
    up_predict = x * (w + lr)
    up_loss = (y - up_predict) ** 2
    
    down_predict = x * (w - lr)
    down_loss = (y - down_predict) ** 2
    
    if up_loss >= down_loss:
        w = w - lr
    else:
        w = w + lr
    
    # Check for improvement
    if loss < best_loss:
        best_loss = loss
        no_improvement = 0
    else:
        no_improvement += 1
        
    # Check if early stopping criteria met
    if no_improvement >= patience:
        print('Early stopping! No improvement for', patience, 'epochs.')
        break
    