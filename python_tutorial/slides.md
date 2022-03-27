# Python Tutorial

---

## What is python

---

## Google Colab

---

## Integrated Development Environment

整合開發環境


---


### Google Colab

- python 3.7.12
- 免費CPU虛擬機
- 可用瀏覽器操作
- 可以與他人共用

---

### 取得教學範例

https://colab.research.google.com/drive/1SnqHaVUk-QjUxi5ieexneYFUoJNrNR2g?usp=sharing

- 先儲存副本：檔案-->在雲端硬碟中儲存副本
![](media/colab_1.png)
- 若尚未登入 google 帳號，會被要求登入

----

## 取得教學範例 

- 視個人情況更改檔名
![](media/colab_2.png)


----

## 執行程式碼
- 輸入 `print("hello world")
- CTRL (或者 command) + ENTER 或者 按下 ▶️
![](media/colab_3.png)
- 注意事項：如果長時間不使用，會被斷線，重新連線之後，寫過的程式碼會被保留，但需要重新執行。
  - 連續使用時間限制：12 小時
  - 閒置時間限制：90 分鐘


----

## 儲存格

![](media/colab_4.png)

----

### 快捷鍵

- 按 <font style= "background:gray">`ESC`</font> 鍵：跳出編輯模式
- 按 <font style= "background:gray">`ENTER`</font> 鍵：進入編輯模式
- <font style= "background:gray">`CTRL`</font> or <font style= "background:gray">`command`</font> + <font style= "background:gray">`ENTER`</font>：執行單一儲存格程式
- <font style= "background:gray">`CTRL`</font> or <font style= "background:gray">`command`</font> + <font style= "background:gray">`M`</font> + <font style= "background:gray">`I`</font>：終止執行中的程式
- 在跳出編輯模式的狀態下：
  - 按 <font style= "background:gray">`A`</font> 鍵：向上新增儲存格
  - 按 <font style= "background:gray">`B`</font> 鍵：向下新增儲存格
  - <font style= "background:gray">`CTRL`</font> or <font style= "background:gray">`command`</font> + <font style= "background:gray">`M`</font> + <font style= "background:gray">`D`</font>：刪除儲存格



---

## 變數

- `x`、`y`、`z` 是變數名稱，19、"Ben"、`True` 為值。
```python
x = 19
y = "Ben"
z = True
```
- 必須以英文字母或者底線作為變數的開頭：data、output、_att
- 英文字母大小寫是不同的：input_1 和 Input_1 是不同變數
- 不可使用保留字：有些字詞本身有功能，不可以當作變數，例如if, and, else, or...等等。

----

## 資料型態

- 數值：`int` 和 `float`
- 字串：`str`
- Boolean：`bool`

----

## 數值運算

```python
num = 12 + 3
print(f"12 + 3 = {num}")
num = 43 - 32
print(f"43 - 32 = {num}")
num = 3 * 8
print(f"3 * 8 = {num}")
num = 3 ** 8
print(f"3 ** 8 = {num}")
num = 19 / 3
print(f"19 / 3 = {num}")
num = 19 % 3
print(f"19 % 3 = {num}")
num = 19 // 3
print(f"19 // 3 = {num}")
```

----

## 數值運算

```
12 + 3 = 15
43 - 32 = 11
3 * 8 = 24
3 ** 8 = 6561
19 / 3 = 6.333333333333333
19 % 3 = 1
19 // 3 = 6
```

----

## 字串

- 字串：以雙引號或單引號匡著文字
```python
string = "青山"
print(string)
```
```
青山
```
- 字串也可以先乘後加
```python
string ="很重要！" * 3 + "所以說三次"
print(string)
```
```
很重要！很重要！很重要！所以說三次
```
- `python`裡，座號是從 0 開始。
```python
print(string[0])
```
```
很
```

----

## 字串

- 換行可以用 `\n`
```python
string = "希望我講課的時候，\n烏克蘭已經得到和平"
print(string)
```
```
希望我講課的時候，
烏克蘭已經得到和平
```
- 如果文字很多段，可以用 3 + 3 個雙引號包起來
```python
string = """
希望我講課的時候，
烏克蘭已經得到和平
"""
print(string)
```

---

## 資料結構

- 列表：
  - `list`
  - `tuple`
- 集合：`set`
- 字典：`dict`



----



## 列表 `list`

- 有序且可變


```python
data = [1, "a", True]
print(data)
```
```
[1, 'a', True]
```
```python
print(data[0])
data[2] = False
print(data[2])
```
```
0
False
```

----

## 列表 `list`
- 用 `list` 和 `range` 產生列表

```python
data = list(range(5))
print(data)
```
```
[0, 1, 2, 3, 4]
```

----

## function of `list` 
- 加一個新的元素
```python
data.append(17)
```
```
[0, 1, 2, 3, 4, 17]
```
- 跟另一個 `list` 合併，增加更多元素
```python
data.extend([10, 67, 23])
```
```
[0, 1, 2, 3, 4, 17, 10, 67, 23]
```
----

## function of `list` 
- 太混亂的話，可以排序一下
```python
data.sort()
```
```
[0, 1, 2, 3, 4, 10, 17, 23, 67]
```
- 把不喜歡的元素移除
```python
data.remove(67)
```
```
[0, 1, 2, 3, 4, 10, 17, 23]
```

----


## 列表 `tuple`


- 有序且不可變
```python
data = (1, "a", True)
print(data)
print(data[0])
```
```
(1, 'a', True)
1
```
- 用 `tuple` 和 `range` 產生列表
```python
data = tuple(range(5))
print(data)
```
```
[0, 1, 2, 3, 4]
```

----

## 列表 `tuple`

- 如果試著改變某個元素

```python
data[2] = False
```
![](media/tuple_error.png)

----

## 列表
- `list` 和 `tuple` 互換

```python
data = list(data)
print(data)
data = tuple(data)
print(tuple)
```
```
[0, 1, 2, 3, 4]
(0, 1, 2, 3, 4)
```

----

## 集合 `set`

```python
data = {12, "a", "b"}
print(data)
```
```
{12, "a", "b"}
```
- 檢查資料是否包含某值

```python
print("a" in data)
```
```
True
```

----

## 集合 `set`
- 交集

```python
data_new = {2, "b", True, False}
print(data & data_new)
```

```
{"b"}
```
- 聯集

```python
print(data | data_new)
```
```
{False, True, 2, 'b', 12, 'a'}
```

- 差集

```python
print(data - data_new)
```
```
{12, 'a'}
```

----

## 字典 `dict`

- key 與 value 的組合
- 可用 key 找到對應的 value
- 只能查詢 key 是否存在於字典中，無法查詢 value
```python
data = {"三月":"March", "四月":"April", "one":1}
print(data["三月"]) # “March"
print("三月" in data) # 印出True
print("March" in data) # 印出False
```
- 也可以用`dict`產生字典
```python
data = dict(三月="March", 四月="April", one=1)
print(data)
```


----

## function of `dict`
```python
print(data.keys())
print(data.values())
print(data.items())
```
```
dict_keys(['三月', '四月', 1])
dict_values(['March', 'April', 'one'])
dict_items([('三月', 'March'), ('四月', 'April'), (1, 'one')])
```

---

```python
type(data)
len(data)
```