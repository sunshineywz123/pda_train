import requests
import json
import urllib
from pyquery import PyQuery as pq
import hashlib
import hmac
from ctypes import *
import threading


def logical_rshift(num, shift):
    """
    算术右移
    :param num: 待移动的数字
    :param shift: 移动的位数
    :return: 移动后的数字
    """
    return c_uint32(num).value >> shift


def info(d, k):

    def _encode(s):

        def _getbyte(s, i):
            x = ord(s[i])
            if x > 255:
                raise ValueError("INVALID_CHARACTER_ERR: DOM Exception 5")
            return x

        _PADCHAR = "="
        _ALPHA = "LVoJPiCN2R8G90yg+hmFHuacZ1OWMnrsSTXkYpUq/3dlbfKwv6xztjI7DeBE45QA"

        s = str(s)
        x = []
        imax = len(s) - len(s) % 3
        if len(s) == 0:
            return s
        for i in range(0, imax, 3):
            b10 = ((_getbyte(s, i) << 16) & 0xffffffff) | (
                (_getbyte(s, i + 1) << 8) & 0xffffffffff) | _getbyte(s, i + 2)
            x.append(_ALPHA[b10 >> 18])
            x.append(_ALPHA[(b10 >> 12) & 63])
            x.append(_ALPHA[(b10 >> 6) & 63])
            x.append(_ALPHA[b10 & 63])
        if len(s) - imax == 1:
            b10 = (_getbyte(s, i) << 16) & 0xffffffff
            x.append(_ALPHA[b10 >> 18] + _ALPHA[(b10 >> 12) & 63] +
                     _PADCHAR * 2)
        elif len(s) - imax == 2:
            b10 = ((_getbyte(s, i) << 16) & 0xffffffff) | (
                (_getbyte(s, i + 1) << 8) & 0xffffffff)
            x.append(_ALPHA[b10 >> 18] + _ALPHA[(b10 >> 12) & 63] +
                     _ALPHA[(b10 >> 6) & 63] + _PADCHAR)
        return "".join(x)

    def xEncode(data, key):

        def str_to_int_array(a, b):
            c = len(a)
            v = []
            for i in range(0, c, 4):
                m1 = a[i]
                m2 = m3 = m4 = 0
                if i + 1 < c:
                    m2 = c_int32(a[i + 1] << 8).value
                if i + 2 < c:
                    m3 = c_int32(a[i + 2] << 16).value
                if i + 3 < c:
                    m4 = c_int32(a[i + 3] << 24).value
                v.append(m1 | m2 | m3 | m4)
            if b:
                v.append(c)
            return v

        def int_array_to_str(a, b):
            d = len(a)
            c = (d - 1) << 2
            if b:
                m = a[d - 1]
                if m < c - 3 or m > c:
                    return None
                c = m
            s = [
                chr(x & 0xff) + chr(logical_rshift(x, 8) & 0xff) +
                chr(logical_rshift(x, 16) & 0xff) +
                chr(logical_rshift(x, 24) & 0xff) for x in a
            ]
            if b:
                return "".join(s)[:c]
            else:
                return "".join(s)

        v = str_to_int_array(data, True)
        k = str_to_int_array(key, False)
        if len(k) < 4:
            k += [0] * (4 - len(k))
        n = len(v) - 1
        z = v[n]
        y = v[0]
        c = c_int32(0x86014019).value | c_int32(0x183639A0).value
        m = 0
        e = 0
        p = 0
        q = int(6 + 52 / (n + 1))
        d = 0
        while q > 0:
            d = c_int32((d + c) & 0xffffffff).value
            e = logical_rshift(d, 2) & 3
            for p in range(n):
                y = v[p + 1]
                m = logical_rshift(
                    z, 5) ^ (c_int32((y << 2) & (0xffffffff)).value)
                m += (logical_rshift(y, 3) ^
                      (c_int32((z << 4) & (0xffffffff)).value)) ^ (d ^ y)
                m += k[(p & 3) ^ e] ^ z
                z = c_int32((v[p] + m) & 0xffffffff).value
                v[p] = z
            p += 1
            y = v[0]
            m = logical_rshift(z, 5) ^ (c_int32((y << 2) & (0xffffffff)).value)
            m += (logical_rshift(y, 3) ^
                  (c_int32((z << 4) & (0xffffffff)).value)) ^ (d ^ y)
            m += k[(p & 3) ^ e] ^ z
            z = c_int32((v[n] + m) & 0xffffffff).value
            v[n] = z
            q -= 1
        return int_array_to_str(v, False)

    encoded_data = json.dumps(d,
                              separators=(',', ':')).encode('unicode-escape')
    encoded_key = k.encode('unicode-escape')
    xencoded_data = xEncode(encoded_data, encoded_key)
    a = _encode(xencoded_data)
    return "{SRBX1}" + _encode(xencoded_data)


def pwd(n, t):
    return hmac.new(t.encode('unicode-escape'),
                    n.encode('unicode-escape'),
                    digestmod=hashlib.md5).hexdigest()


def chksum(d):
    return hashlib.sha1(d.encode('unicode-escape')).hexdigest()


class ZJUWLAN(object):

    def __init__(self, username, password, PORTAL_URL):
        self.username = username
        self.password = password
        self.PORTAL_URL = PORTAL_URL
        self.header = {
            'User-Agent':
            'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1847.116 Safari/537.36'
        }

    def login(self):
        '''
        Try logging in ZJUWLAN. Return True if successful.
        '''

        response = pq(url=self.PORTAL_URL)
        self.ip = response('#user_ip').attr('value')
        self.ac_id = response('#ac_id').attr('value')
        sess = requests.Session()
        challenge_form = {
            "callback": "jQuery1124012077694741281753_1687174585518",
            "username": self.username,
            "ip": self.ip,
        }
        challenge_url = self.PORTAL_URL + "/cgi-bin/get_challenge?" + urllib.parse.urlencode(
            challenge_form)
        # print(challenge_url)
        r = sess.get(url=challenge_url, headers=self.header)
        print(r.text)
        challenge_dict = eval(
            r.text.split("(", maxsplit=1)[-1].rsplit(")", maxsplit=2)[0])
        # print(challenge_dict)

        token = challenge_dict["challenge"]
        self.ip = challenge_dict["client_ip"]
        enc = "srun_bx1"
        i = info(
            {
                "username": self.username,
                "password": self.password,
                "ip": self.ip,
                "acid": self.ac_id,
                "enc_ver": enc,
            }, token)
        hmd5 = pwd(self.username, token)

        n = '200'
        type = '1'
        chkstr = token + self.username
        chkstr += token + hmd5
        chkstr += token + self.ac_id
        chkstr += token + self.ip
        chkstr += token + n
        chkstr += token + type
        chkstr += token + i

        self.password = "{MD5}" + hmd5

        print("Try logging with username {}".format(self.username))
        login_form = {
            "callback": "jQuery1124012077694741281753_1687174585518",
            "action": "login",
            "username": self.username,
            "password": self.password,
            "ac_id": self.ac_id,
            "ip": self.ip,
            "chksum": chksum(chkstr),
            "info": i,
            "n": n,
            "type": type,
        }

        login_url = self.PORTAL_URL + "/cgi-bin/srun_portal?" + urllib.parse.urlencode(
            login_form)
        r = requests.get(login_url, headers=self.header)
        # print(r.text)
        # print("First login reply: {}".format(r.text))
        if "login_ok" in r.text:
            print('Login successfully')
        elif "\"error\":\"ok\"" in r.text:
            print('Already logged in')
        else:
            print("Login error")
        return r.text
import sys
sys.path.append('.')
from lib.utils.dingtalk.robot import send_message

interval = 86400

def test():
    username = '12321131'
    password = 'a9dhb5xta6Lht'
    PORTAL_URL = "https://net.zju.edu.cn"
    app = ZJUWLAN(username, password, PORTAL_URL)
    json_str = app.login()
    clean_json_str = json_str.replace('jQuery1124012077694741281753_1687174585518(', '').rstrip(')')
    data = json.loads(clean_json_str)
    message = '{}: {} \n{}: {}'.format('client_ip', data.get('client_ip'), 
                                       'online_ip', data.get('online_ip'))
    send_message(message)
    start_timer(interval)
    

def start_timer(interval):
    # Create a Timer object that will call my_function after 'interval' seconds
    timer = threading.Timer(interval, test)
    timer.start()
    return timer    

def main():
    # Create a Timer object that will call my_function after 'interval' seconds
    start_timer(interval)
    
    
if __name__ == '__main__':
    main()