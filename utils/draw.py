import matplotlib.pyplot as plt


def drawLoss(loss, output: str, file='train_loss'):
    #对测试Loss进行可视化
    plt.plot(loss, label=file)
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('Loss')
    # plt.show()
    plt.savefig(output+file+'.png')
    plt.show()
    plt.close()