# 不文明用语分类任务对抗样本生成

### 示例
`你是大笨蛋` --> `拟是大笨，蛋`

### 混淆方式
- [x] 形近字替换
- [x] 音近字替换
- [x] 标点插入
- [x] shuffle
- [x] 生僻字删除
- [x] 追加文明用语
- [x] 以上综合
### 生成方式
- [x] 重要性评估
    - [x] 删除评估
    - [x] 替换评估
    - [x] 头评估
    - [x] 尾评估
    - [x] 以上综合
- [x] 规则式
### 分类模型
- [x] textcnn
- [x] word2vec+lightGBM
- [x] fastText
- [x] Transformer
- [x] CNN
- [X] RNN 
- [x] TFIDF+Classifier
- [x] TFIDF+word2vec+Classifier
### 评估方式
- [x] 上述所有模型和攻击方式在n折交叉验证下准确率等指标的变化情况
- [x] 文本变化距离

