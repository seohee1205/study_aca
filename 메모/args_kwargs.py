2023 04 27 목

*args
**params
: 함수 인자를 정의하는 데 사용되는 특별한 매개변수

*args는 함수를 호출할 때 인자를 튜플 형태로 받아서 함수 내에서 사용할 수 있게 함
args는 arguments의 약자, *는 가변 인자를 의미함
ex)
def foo(*args):
     print(args)
foo(1, 2, 3)	# (1, 2, 3) 출력

**kwargs는 함수를 호출할 때 키워드 인자를 딕셔너리 형태로 받아서 함수 내에서 사용할 수 있게 함
kwargs는 keyword arguments의 약자, **는 가변 키워드 인자를 의미함
ex)
def bar(**kwargs):
     print(kwargs)