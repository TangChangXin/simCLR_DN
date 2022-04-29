import torch, argparse, os, random
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import SimCLRModel
from tqdm import tqdm
from torch.backends.cudnn import deterministic


class 无标签眼底图像数据集(torch.utils.data.Dataset):
    # 读取图像后对每一张图像进行两种不同的随机图像变换并返回两个变换结果，作为正对样本输入
    def __init__(self, 文件路径, 图像变换):
        self.文件路径 = 文件路径 # 保存图像的文件目录
        self.图像变换 = 图像变换
        self.图像列表 = os.listdir(self.文件路径) # 得到图像名称列表

    def __len__(self):
        return len(self.图像列表) # 返回数据集的大小

    def __getitem__(self, item):
        # 根据item返回对应的图像
        图像名 = self.图像列表[item] # 根据索引取出对应的图像
        图像路径 = os.path.join(self.文件路径, 图像名) # 得到图像的路径
        图像 = Image.open(图像路径).convert('RGB') # todo 三通道读取吗？读取图像并转为RGB格式

        # 对同一图片应用不同图像变换得到两个结果，作为正对样本输入
        图像变换结果1 = self.图像变换(图像)
        图像变换结果2 = self.图像变换(图像)
        return 图像变换结果1, 图像变换结果2


def 训练模型(网络模型, 优化器, 硬件设备, 命令行参数, 随机图像变换):
    # TODO 定义数据集路径，选择用不同的数据
    无标签训练数据集 = 无标签眼底图像数据集(文件路径='../Dataset/OCTA_6M/Projection Maps/OCTA(FULL)', 图像变换=随机图像变换["训练集"])
    # win可能多线程报错，num_workers最多和CPU的超线程数目相同，若报错设为0
    # todo 如果维度报错注意修改drop_last
    训练数据 = torch.utils.data.DataLoader(无标签训练数据集, batch_size=命令行参数.unlabeled_data_batch_size, shuffle=True, num_workers=6, drop_last=True, pin_memory=True)
    # 训练数据 = torch.utils.data.DataLoader(无标签训练数据集, batch_size=命令行参数.unlabeled_data_batch_size, shuffle=True, num_workers=2, drop_last=False, pin_memory=True)

    训练损失函数 = SimCLRModel.对比损失函数().to(硬件设备)  # 使用余弦相似度损失函数
    最佳损失 = float("inf") # 初始无穷大,训练时损失计算到15位小数
    学习率调整器 = torch.optim.lr_scheduler.ReduceLROnPlateau(优化器, mode='min', factor=0.5, patience=120, cooldown=10)
    for 当前训练周期 in range(1, 命令行参数.unlabeled_train_max_epoch):
        网络模型.train()  # 开始训练
        当前训练周期全部损失 = 0
        # 每一批数据训练。enumerate可以在遍历元素的同时输出元素的索引
        训练循环 = tqdm(enumerate(训练数据), total=len(训练数据), ncols=130, colour='#d33682', leave=True)
        for 训练批次, (图像变换1, 图像变换2) in 训练循环:
            图像变换1, 图像变换2 = 图像变换1.to(硬件设备), 图像变换2.to(硬件设备)

            _, 特征1 = 网络模型(图像变换1)  # 特征1是最终输出特征 形状[批量大小 * 特征维度]
            _, 特征2 = 网络模型(图像变换2)  # 特征2是最终输出特征 形状[批量大小 * 特征维度]

            # 计算特征1和特征2之间的余弦相似度
            训练损失 = 训练损失函数(特征1, 特征2, 命令行参数.unlabeled_data_batch_size)
            优化器.zero_grad()
            训练损失.backward()
            优化器.step()
            当前训练周期全部损失 += 训练损失.detach().item()
            训练循环.set_description(f'训练周期 [{当前训练周期}/{命令行参数.unlabeled_train_max_epoch}]') # 设置进度条标题
            训练循环.set_postfix(训练损失 = 训练损失.detach().item()) # 每一批训练都更新损失

        # 参数'a',打开一个文件用于追加。若该文件已存在，文件指针将会放在文件的结尾，新的内容将会被写入到已有内容之后。若该文件不存在，创建新文件进行写入。
        with open(os.path.join("Weight", "OCTA(FULL)" + 优化器.__class__.__name__ + "unlabeledloss.txt"), 'a') as f:
            # 将损失写入文件并分隔
            f.write(str(当前训练周期全部损失) + '\n')

        学习率调整器.step(当前训练周期全部损失)  # 调整学习率
        with open(os.path.join("Weight", "OCTA(FULL)" + 优化器.__class__.__name__ + "unlabeledlearnrate.txt"), 'a') as f:
            # 记录学习率
            f.write(str(优化器.param_groups[0]['lr']) + '\n')

        if 当前训练周期全部损失 < 最佳损失:
            最佳损失 = 当前训练周期全部损失
            # 保存模型
            torch.save(网络模型.state_dict(), os.path.join("Weight", "OCTA(FULL)" + 优化器.__class__.__name__ + "unlabeledBest_model" + ".pth"))


# 第一阶段无标签训练
def 无标签训练(命令行参数):
    全部随机数种子 = 222 # init seed 初始化随机种子
    # 下面似乎都是控制生成相同的随机数
    random.seed(全部随机数种子)
    np.random.seed(全部随机数种子)
    torch.manual_seed(全部随机数种子)
    torch.cuda.manual_seed_all(全部随机数种子)
    torch.cuda.manual_seed(全部随机数种子)
    np.random.seed(全部随机数种子)
    os.environ['PYTHONHASHSEED'] = str(全部随机数种子) # 禁止哈希随机化，使实验可复现

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
            transforms.RandomResizedCrop(命令行参数.unlabeled_train_resize),  # 随机选取图像中的某一部分然后再缩放至指定大小
            transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
            # 修改亮度、对比度和饱和度
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # 随机应用添加的各种图像变换
            transforms.RandomGrayscale(p=0.2),  # todo 随机灰度化，但我本来就是灰度图啊
            transforms.ToTensor(),  # 转换为张量且维度是[C, H, W]
            # 三通道归一化
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    }

    '''
    # 用于无标签数据训练的模型
    网络模型 = SimCLRModel.无监督simCLRresnet50()
    残差网络预训练权重路径 = "Weight/resnet50-19c8e357.pth"
    assert os.path.exists(残差网络预训练权重路径), "残差模型权重{}不存在.".format(残差网络预训练权重路径)
    残差模型参数 = torch.load(残差网络预训练权重路径, map_location=硬件设备) # 字典形式读取Res50的权重
    simCLR模型参数 = 网络模型.state_dict() # 自己设计的模型参数字典

    # 我的模型只使用Res50模型从开头的第一个卷积层到最后一个全局自适应池化层作为编码器，所以遍历Res50的参数并赋值给我模型中对应名称的参数
    编码器参数 = {键:值 for 键,值 in 残差模型参数.items() if 键 in simCLR模型参数.keys()}
    simCLR模型参数.update(编码器参数) # 更新我模型的参数，实际就是使用Res50模型参数
    网络模型.load_state_dict(simCLR模型参数) # 加载模型参数
    '''

    网络模型1 = SimCLRModel.无监督simCLR50(True)
    网络模型1.to(硬件设备)
    优化器1 = torch.optim.Adam(网络模型1.parameters())
    训练模型(网络模型1, 优化器1, 硬件设备, 命令行参数, 随机图像变换)

    网络模型2 = SimCLRModel.无监督simCLR50(True)
    网络模型2.to(硬件设备)
    优化器2 = torch.optim.AdamW(网络模型2.parameters())
    训练模型(网络模型2, 优化器2, 硬件设备, 命令行参数, 随机图像变换)

    网络模型3 = SimCLRModel.无监督simCLR50(True)
    网络模型3.to(硬件设备)
    优化器3 = torch.optim.Adamax(网络模型3.parameters())
    训练模型(网络模型3, 优化器3, 硬件设备, 命令行参数, 随机图像变换)

    网络模型4 = SimCLRModel.无监督simCLR50(True)
    网络模型4.to(硬件设备)
    优化器4 = torch.optim.Adadelta(网络模型4.parameters())
    训练模型(网络模型4, 优化器4, 硬件设备, 命令行参数, 随机图像变换)


if __name__ == '__main__':
    # 设置一个参数解析器
    命令行参数解析器 = argparse.ArgumentParser(description='无标签数据训练 SimCLR')
    # 添加无标签数据训练时的参数
    命令行参数解析器.add_argument('--unlabeled_data_batch_size', default=2, type=int, help="无标签数据训练时的批量大小")
    命令行参数解析器.add_argument('--unlabeled_train_max_epoch', default=6000, type=int, help="无标签训练的最大迭代周期")
    命令行参数解析器.add_argument('--unlabeled_train_resize', default=224, type=int, help="随机缩放图像的大小")
    # 获取命令行传入的参数
    无标签训练命令行参数 = 命令行参数解析器.parse_args()
    无标签训练(无标签训练命令行参数)