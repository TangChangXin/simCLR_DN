#  有标签数据
创建LabeledDataset文件夹，其中再分别创建Train和Validate文件夹，用来保存训练数据和验证数据。
Train和Validate文件夹中需要按照不同类别分别创建保存不同类别图像的文件夹。
```
>LabeledDataset
    >train
        >DN
            >图像1.jpg
            >图像2.jpg
        >NDN
            >图像1.jpg
            >图像1.jpg
        >...
    >Validate
        >DN
            >图像1.jpg
            >图像2.jpg
        >NDN
            >图像1.jpg
            >图像2.jpg
        >...
```
#  无标签数据
直接存放OCTA-500的数据
#  模型权重
ResNet50预训练权重和自己的权重都保存在Weight中