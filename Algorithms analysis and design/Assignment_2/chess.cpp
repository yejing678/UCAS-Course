#include <iostream>
#include <algorithm>
#include <cstdio>
 
using namespace std;
int box[1500][1500]; // 2的10次幂为1024
int m = 1;  //骨牌编号
void whatf(int x, int y, int a, int b, int length);
 
int main()
{
    int x, y, length;
    cin >> x >> y >> length;
    x--;y--;
 
    whatf(x, y, 0, 0, length); //按照提议的函数
 
    for(int i = 0; i < length; i++) //输出答案
    {
        for(int j = 0; j < length; j++)
            printf("%4d", box[i][j]);
        cout << endl;
    }
 
    return 0;
 
}
 
/*让没有被操作过的方格设为-1
分为几种情况，当只有一个时，就直接是0
当有length = 2时，判断一下哪个被修改过，将其余三个改成m，m再自增
当length > 2时，就将拆分成4个部分，将除了含有小红点的部分都改为m
*/
 
 
 
//x，y为特殊方块的位置，a，b为长和宽的起点，length为正方形尺寸
void whatf(int x, int y, int a, int b, int length)
{
    if(length == 1)
    {
        return;
    }
    int title = m++;
    int size = length/2;
    //如果特殊方块在左上
    if(x < a+size && y < b+size)
    {
        whatf(x, y, a, b, size);
    }
    else //特殊方块不在左上的部分，就将该部分的右下角置为当前骨牌序号
    {
        box[a+size-1][b+size-1] = title; //将右下角置为当前骨牌序号
        whatf(a+size-1, b+size-1, a, b, size);
    }
 
    //如果特殊方块在右上
    if(x < a+size && y >= b+size)
    {
        whatf(x, y, a, b+size, size);
    }
    else//特殊方块不在右上的部分，就将该部分的左下角置为当前骨牌序号
    {
        box[a+size-1][b+size] = title; //将左下角置为当前骨牌序号
        whatf(a+size-1, b+size, a, b+size, size);
    }
 
    //如果特殊方块在右下
    if(x >= a+size && y >= b+size)
    {
        whatf(x, y, a+size, b+size, size);
    }
    else
    {
        box[a+size][b+size] = title;
        whatf(a+size, b+size, a+size, b+size, size);
    }
 
    //如果特殊方块在左下
    if(x >= a+size && y < b+size)
    {
        whatf(x, y, a+size, b, size);
    }
    else
    {
        box[a+size][b+size-1] = title;
        whatf(a+size, b+size-1, a+size, b, size);
    }
}
