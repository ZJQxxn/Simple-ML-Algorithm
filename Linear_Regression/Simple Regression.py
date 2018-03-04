import matplotlib.pyplot as plt
import matplotlib.patches as patches


def linear_regression(data):
    #initial w and b
    w=0
    b=0
    learning_rate = 0.0001

    epoch=1
    while epoch<=100:
        w,b=compute_gradient(data,w,b,learning_rate)
        epoch+=1
    print(w,b)
    return w,b


def compute_gradient(data,current_w,current_b,learning_rate):
    w=current_w
    b=current_b
    #gradient descent
    w_gradient = 0.0
    b_gradient = 0.0
    for index in range(len(data[0])):
        x = data[0][index]
        y = data[1][index]
        w_gradient += (-x) * (y - (w * x + b))
        b_gradient += (-(y - (w * x + b)))

    # print("Epoch %d, w_gra:%f  b_gra:%f"%(epoch,w_gradient,b_gradient))
    current_w = w - learning_rate * w_gradient
    current_b = b - learning_rate * b_gradient
    return current_w,current_b


def showData(data1,data2):
    plt.plot(data1[0],data1[1],'ro-',data2[0],data2[1],'b*-')
    plt.title('Y-X')
    plt.xlabel('X')
    plt.ylabel('Y')
    red_patch=patches.Patch(color='red',label='Original Data')
    blue_patch=patches.Patch(color='blue',label='Estimated Data')
    plt.legend(handles=[red_patch,blue_patch])
    plt.show()



if __name__ == '__main__':
    X=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    Y=[1.1,2.2,3.3,4.4,5.5,6.6,7.7,8.8,9.9,10.1,11.11,12.12,13.13,14.14,15.15]
    data1=(X,Y)

    w,b=linear_regression(data1)
    Y_Esitimate=[w*x+b for x in X]

    data2=(X,Y_Esitimate)
    showData(data1,data2)


