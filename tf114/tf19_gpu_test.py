import tensorflow as tf

# tf.compat.v1.disable_eager_execution()  # 즉시 실행모드 안 해 (1.0 쓸 거야)
# tf.compat.v1.enable_eager_execution()   # 즉시 실행모드 해 (2.0 쓸 거야)

print("텐서플로 버전 : ", tf.__version__)   # 1.14.0
print("즉시실행모드 : ", tf.executing_eagerly())    # False

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        print(gpus[0])
    except RuntimeError as e:
        print(e)
        
else : 
    print("gpu 없다!")


# 274
# True, disable: False, enable: True
# False, disable: False, enable: True

