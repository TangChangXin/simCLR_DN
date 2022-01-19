import torch, argparse, os, random
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import numpy as np
import SimCLRModel
from tqdm import tqdm
from torch.backends.cudnn import deterministic


def 训练模型(网络模型, 优化器, 预训练权重路径, 硬件设备, 命令行参数, 随机图像变换):
    # 加载训练数据集和测试数据集
    数据集根路径 = "LabeledDataset"
    有标签训练数据集 = datasets.ImageFolder(root=os.path.join(数据集根路径, "Train"), transform=随机图像变换["测试集"])
    有标签训练数据 = torch.utils.data.DataLoader(有标签训练数据集, batch_size=命令行参数.labeled_data_batch_size, shuffle=True,
                                          num_workers=0, pin_memory=True)
    有标签测试数据集 = datasets.ImageFolder(root=os.path.join(数据集根路径, "Validate"), transform=随机图像变换["测试集"])
    有标签测试数据 = torch.utils.data.DataLoader(有标签测试数据集, batch_size=命令行参数.labeled_data_batch_size, shuffle=False,
                                          num_workers=0, pin_memory=True)
    assert os.path.exists(预训练权重路径), "自监督预训练模型权重{}不存在.".format(预训练权重路径)
    网络模型.load_state_dict(torch.load(预训练权重路径), strict=False)  # 加载无监督预训练模型
    网络模型.to(硬件设备)
    训练损失函数 = torch.nn.CrossEntropyLoss()
    测试损失函数 = torch.nn.CrossEntropyLoss()
    学习率调整器 = torch.optim.lr_scheduler.ReduceLROnPlateau(优化器, mode='max', factor=0.5, patience=10, cooldown=10)
    最高测试准确率 = 0.0
    # 开始训练
    for 当前训练周期 in range(1, 命令行参数.labeled_train_max_epoch + 1):
        网络模型.train()
        当前周期全部训练损失 = 0.0
        # 每一批数据训练。enumerate可以在遍历元素的同时输出元素的索引
        训练循环 = tqdm(enumerate(有标签训练数据), total=len(有标签训练数据), leave=True)
        for 当前批次, (图像数据, 标签) in 训练循环:
            图像数据, 标签 = 图像数据.to(硬件设备), 标签.to(硬件设备)
            训练集预测概率 = 网络模型(图像数据)
            训练损失 = 训练损失函数(训练集预测概率, 标签)  # 每一批的训练损失
            优化器.zero_grad()
            训练损失.backward()
            优化器.step()
            当前周期全部训练损失 += 训练损失.detach().item()
            训练循环.desc = "训练迭代周期 [{}/{}] 当前训练损失：{:.8f}".format(当前训练周期, 命令行参数.labeled_train_max_epoch,
                                                            训练损失.detach().item())  # 设置进度条描述

        学习率调整器.step(当前周期全部训练损失)  # 调整学习率
        # 记录每个周期训练损失值
        with open(os.path.join("Weight", "OCTA(FULL)" + 优化器.__class__.__name__ + "LabeledTrainLoss.txt"), "a") as f:
            f.write(str(当前周期全部训练损失) + "\n")


        网络模型.eval() # 每一周期数据训练完成后测试模型效果
        当前周期全部测试损失 = 0.0
        测试正确的总数目 = 0
        # 下方代码块不反向计算梯度
        with torch.no_grad():
            测试循环 = tqdm(enumerate(有标签测试数据), total=len(有标签测试数据), leave=True)
            for 当前批次, (图像数据, 标签) in 测试循环:
                图像数据, 标签 = 图像数据.to(硬件设备), 标签.to(硬件设备)
                测试集预测概率 = 网络模型(图像数据)
                测试损失 = 测试损失函数(测试集预测概率, 标签)
                当前周期全部测试损失 += 测试损失.detach().item()
                # torch.max(a,1)返回行最大值和列索引。结果中的第二个张量是列索引
                预测类别 = torch.max(测试集预测概率, dim=1)[1]
                测试正确的总数目 += torch.eq(预测类别, 标签).sum().item()  # 累加每个批次预测正确的数目
                测试循环.desc = "测试迭代周期 [{}/{}] 当前测试损失：{:.8f}".format(当前训练周期, 命令行参数.labeled_train_max_epoch, 测试损失.detach().item())  # 设置进度条描述
        当前迭代周期测试准确率 = 测试正确的总数目 / len(有标签测试数据集)

        # 记录每个周期测试损失值
        with open(os.path.join("Weight", "OCTA(FULL)" + 优化器.__class__.__name__ + "LabeledTestLoss.txt"), "a") as f:
            f.write(str(当前周期全部测试损失) + "\n")

        with open(os.path.join("Weight", "OCTA(FULL)" + 优化器.__class__.__name__ + "TestAccuracy.txt"), "a") as f:
            f.write(str(当前迭代周期测试准确率) + "\n")

        # 以最高测试准确率作为模型保存的标准
        if 当前迭代周期测试准确率 > 最高测试准确率:
            最高测试准确率 = 当前迭代周期测试准确率
            torch.save(网络模型.state_dict(), os.path.join("Weight", "OCTA(FULL)" + 优化器.__class__.__name__ + "BestLabeledModel" + ".pth"))


def 有标签训练(命令行参数):
    #  init seed 初始化随机种子
    全部随机数种子 = 222

    # 下面似乎都是控制生成相同的随机数
    random.seed(全部随机数种子)
    np.random.seed(全部随机数种子)  # todo
    torch.manual_seed(全部随机数种子)
    torch.cuda.manual_seed_all(全部随机数种子)
    torch.cuda.manual_seed(全部随机数种子)
    np.random.seed(全部随机数种子)  # todo

    # 禁止哈希随机化，使实验可复现
    os.environ['PYTHONHASHSEED'] = str(全部随机数种子)

    # 设置训练使用的设备
    if torch.cuda.is_available():
        硬件设备 = torch.device("cuda:0")
        # 保证每次返回的卷积算法将是确定的，如果配合上设置 Torch 的随机种子为固定值的话，应该可以保证每次运行网络的时候相同输入的输出是固定的。
        torch.backends.cudnn.deterministic = True
        if torch.backends.cudnn.deterministic:
            print("确定卷积算法")
        torch.backends.cudnn.benchmark = False  # 为每层搜索适合的卷积算法实现，加速计算
    else:
        硬件设备 = torch.device("cpu")
    print("训练使用设备", 硬件设备)

    随机图像变换 = {
        "训练集": transforms.Compose([
            # TODO 选择合适的图像大小。是否需要随机高斯滤波
            transforms.RandomResizedCrop(命令行参数.labeled_train_resize),  # 随机选取图像中的某一部分然后再缩放至指定大小
            transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
            # 修改亮度、对比度和饱和度
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # 随机应用添加的各种图像变换
            transforms.RandomGrayscale(p=0.2),  # todo 随机灰度化，但我本来就是灰度图啊
            transforms.ToTensor(),  # 转换为张量且维度是[C, H, W]
            # 三通道归一化
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]),
        "测试集": transforms.Compose([
            transforms.ToTensor(),
            # 三通道归一化
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    }

    # 优化器 = torch.optim.Adam(分类模型.全连接.parameters(), lr=1e-3, weight_decay=1e-6) # 只训练分类层
    分类模型1 = SimCLRModel.有监督simCLRresnet50(2)
    优化器1 = torch.optim.Adam(分类模型1.parameters())
    训练模型(分类模型1, 优化器1, "Weight/OCTA(FULL)AdamunlabeledBest_model.pth",硬件设备, 命令行参数, 随机图像变换)

    # 分类模型2 = SimCLRModel.有监督simCLRresnet50(2)
    # 优化器2 = torch.optim.Adam(分类模型2.parameters())
    # 训练模型(分类模型2, 优化器2, 硬件设备, 命令行参数, 随机图像变换)

    # 分类模型3 = SimCLRModel.有监督simCLRresnet50(2)
    # 优化器3 = torch.optim.Adam(分类模型3.parameters())
    # 训练模型(分类模型3, 优化器3, 硬件设备, 命令行参数, 随机图像变换)

    # 分类模型4 = SimCLRModel.有监督simCLRresnet50(2)
    # 优化器4 = torch.optim.Adam(分类模型4.parameters())
    # 训练模型(分类模型4, 优化器4, 硬件设备, 命令行参数, 随机图像变换)

    # 开始训练


if __name__ == '__main__':
    # 设置一个参数解析器
    命令行参数解析器 = argparse.ArgumentParser(description="有标签训练 SimCLR")

    # 添加有标签数据训练时的参数
    命令行参数解析器.add_argument('--labeled_data_batch_size', default=1, type=int, help="有标签数据训练时的批量大小")
    命令行参数解析器.add_argument('--labeled_train_max_epoch', default=2, type=int, help="有标签训练的最大迭代周期")
    命令行参数解析器.add_argument('--labeled_train_resize', default=224, type=int, help="随机缩放图像的大小")

    # 获取命令行传入的参数
    有标签训练命令行参数 = 命令行参数解析器.parse_args()
    有标签训练(有标签训练命令行参数)
