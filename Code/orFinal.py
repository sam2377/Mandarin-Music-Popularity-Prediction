import pulp
if __name__ == "__main__":
    print("YA")
    model = pulp.LpProblem("valueMax", pulp.LpMaximize) 
    x = pulp.LpVariable('x',lowBound = 0, cat='Binary')  #  pulp.LpVariable()加入變數
    y = pulp.LpVariable('y',lowBound = 0, cat='Binary')
    z = pulp.LpVariable('z',lowBound = 0, cat='Binary')

    # model += 設置目標函數
    model += x+y+2*z

    # model += 加入限制式
    model += x+2*y+3*z <= 4
    model += x+y >= 1

    model.solve()  # mmodel.solve()求解

    # 透過屬性varValue,name顯示決策變數名字及值
    for v in model.variables():
        print(v.name, "=", v.varValue)

    # 透過屬性value(model.objective)顯示最佳解
    print('obj=', pulp.value(model.objective))