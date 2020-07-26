#实验设备
- 小米aqara网关
- 小米空调伴侣（支持网关）
- 小米温湿度传感器

#环境配置
- Homeassistant 环境安装
  
  官网地址：https://www.hachina.io/docs/2104.html
  
  更改数据库：
     -  安装mysql数据库
     -  创建数据库hass
     -  修改configuration配置文件，加入如下命令
    ```
         recorder:
            db_url: mysql+pymysql://user:passwd@SERVER_IP/DB_NAME?charset=utf8
          
    ```     
     - 删除 .homeassistant 下面的 home-assistant_v2.db文件  
     
     - 重启 homeassistant
     
       
       
     
