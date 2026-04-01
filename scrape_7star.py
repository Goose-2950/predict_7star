"""
7星彩历史开奖号码爬取 v2.0
═══════════════════════════
修复: 第7位范围 0~14（可为两位数）

依赖: pip install requests
"""

import requests
import json
import re
import csv
import time
import random


# ============================================================
#  核心修复：七星彩号码解析器
# ============================================================
def parse_lottery_numbers(raw_str):
    """
    智能解析七星彩号码
    ─────────────────────
    规则: 位置1~6 范围 0~9（单位数）
          位置7   范围 0~14（可两位数）

    支持的输入格式:
      "3 9 9 1 5 3 14"       → 空格分隔
      "3,9,9,1,5,3,14"       → 逗号分隔
      "399153 14"             → 前6位拼接 + 空格 + 第7位
      "39915314"              → 8位拼接(最后两位是第7位>9时)
      "3991535"               → 7位拼接(第7位<=9时)

    返回: ['3','9','9','1','5','3','14'] 或 None
    """
    if not raw_str:
        return None

    s = str(raw_str).strip()

    # ── 方法1: 按分隔符拆分 ──
    # 把各种分隔符统一为空格
    normalized = re.sub(r'[,;|+\s]+', ' ', s).strip()
    parts = normalized.split()

    if len(parts) == 7:
        # 完美：7个独立的数字
        try:
            nums = [int(p) for p in parts]
            if (all(0 <= nums[i] <= 9 for i in range(6))
                    and 0 <= nums[6] <= 14):
                return [str(n) for n in nums]
        except ValueError:
            pass

    # ── 方法2: 提取所有完整数字 ──
    all_nums = re.findall(r'\d+', s)

    if len(all_nums) == 7:
        try:
            nums = [int(n) for n in all_nums]
            if (all(0 <= nums[i] <= 9 for i in range(6))
                    and 0 <= nums[6] <= 14):
                return [str(n) for n in nums]
        except ValueError:
            pass

    # ── 方法3: 两段式 "399153 14" 或 "399153" + "5" ──
    if len(all_nums) == 2:
        front_str, back_str = all_nums
        if len(front_str) == 6 and len(back_str) <= 2:
            try:
                first6 = [int(c) for c in front_str]
                last1 = int(back_str)
                if (all(0 <= d <= 9 for d in first6) and 0 <= last1 <= 14):
                    return [str(d) for d in first6] + [str(last1)]
            except ValueError:
                pass

    # ── 方法4: 单个拼接字符串 ──
    if len(all_nums) == 1:
        digit_str = all_nums[0]

        # 7位: 所有位置都是0~9
        if len(digit_str) == 7:
            nums = [int(c) for c in digit_str]
            if all(0 <= n <= 9 for n in nums):
                return [str(n) for n in nums]

        # 8位: 前6位单位数 + 最后2位是第7位(10~14)
        if len(digit_str) == 8:
            first6 = [int(c) for c in digit_str[:6]]
            last1 = int(digit_str[6:])
            if (all(0 <= d <= 9 for d in first6) and 10 <= last1 <= 14):
                return [str(d) for d in first6] + [str(last1)]

    # ── 方法5: 多于7个独立数字 ──
    # 可能是带额外信息（中奖金额等混在一起），取前7个验证
    if len(all_nums) > 7:
        try:
            nums = [int(n) for n in all_nums[:7]]
            if (all(0 <= nums[i] <= 9 for i in range(6))
                    and 0 <= nums[6] <= 14):
                return [str(n) for n in nums]
        except ValueError:
            pass

    return None


def parse_front_back(front_str, back_str):
    """
    解析前区+后区格式
    前区(6个0-9数字) + 后区(1个0-14数字)
    """
    if not front_str:
        return None

    front = str(front_str).strip()
    back = str(back_str).strip() if back_str else ''
    combined = front + ' ' + back

    return parse_lottery_numbers(combined)


def parse_number_arrays(front_arr, back_arr):
    """
    解析前区数组 + 后区数组
    例: [3,9,9,1,5,3] + [14] → ['3','9','9','1','5','3','14']
    """
    front = front_arr if isinstance(front_arr, list) else []
    back = back_arr if isinstance(back_arr, list) else []
    all_nums = [int(n) for n in front + back]

    if len(all_nums) == 7:
        if (all(0 <= all_nums[i] <= 9 for i in range(6))
                and 0 <= all_nums[6] <= 14):
            return [str(n) for n in all_nums]

    return None


# ============================================================
#  爬取主函数
# ============================================================
def scrape_7star():
    """直接请求真实API爬取7星彩全部历史开奖号码"""

    print("=" * 70)
    print("  7星彩历史开奖号码爬取 v2.0")
    print("  修复: 第7位范围0~14（支持两位数）")
    print("=" * 70)

    base_url = "https://jc.zhcw.com/port/client_json.php"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                       '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Referer': 'https://www.zhcw.com/kjxx/xqxc/',
    }

    all_data = []

    # ================================================================
    # 策略1：直接请求大量期数
    # ================================================================
    print("\n[1] 尝试一次性请求全部数据...")

    for issue_count in [5000, 3000, 2000, 1000]:
        ts = int(time.time() * 1000)
        callback_name = f"jQuery_{ts}"

        params = {
            'callback': callback_name,
            'transactionType': '10001001',
            'lotteryId': '287',
            'issueCount': str(issue_count),
            'startIssue': '',
            'endIssue': '',
            'startDate': '',
            'endDate': '',
            'type': '0',
            'pageNum': '1',
            'pageSize': str(issue_count),
            'tt': str(random.random()),
            '_': str(ts),
        }

        print(f"    请求 issueCount={issue_count} ...", end=" ")
        try:
            resp = requests.get(base_url, params=params, headers=headers, timeout=30)
            text = resp.text.strip()
            print(f"响应 {len(text)} 字节")

            data = parse_jsonp(text)
            if data is None:
                print(f"    ⚠ 解析失败")
                continue

            with open("raw_api_response.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"    ✓ 已保存原始JSON到 raw_api_response.json")

            print(f"    数据结构:")
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, list):
                        print(f"      '{k}': list[{len(v)}]")
                        if v and isinstance(v[0], dict):
                            print(f"        [0] keys: {list(v[0].keys())}")
                            print(f"        [0] = {json.dumps(v[0], ensure_ascii=False)[:300]}")
                        elif v:
                            print(f"        [0] = {str(v[0])[:100]}")
                    else:
                        print(f"      '{k}': {type(v).__name__} = {str(v)[:100]}")

            records = extract_records(data)
            if records:
                print(f"    ✓ 解析出 {len(records)} 条记录!")
                # 验证报告
                validate_records(records)
                all_data = records
                break
            else:
                print(f"    未能解析出号码记录，尝试下一个参数...")

        except requests.exceptions.Timeout:
            print("超时")
        except Exception as e:
            print(f"失败: {e}")

    # ================================================================
    # 策略2：分页请求
    # ================================================================
    if not all_data:
        print("\n[2] 尝试分页请求...")
        page = 0
        consecutive_empty = 0

        while page < 200:
            page += 1
            ts = int(time.time() * 1000)

            params = {
                'callback': f'jQuery_{ts}',
                'transactionType': '10001001',
                'lotteryId': '287',
                'issueCount': '30',
                'startIssue': '',
                'endIssue': '',
                'startDate': '',
                'endDate': '',
                'type': '0',
                'pageNum': str(page),
                'pageSize': '30',
                'tt': str(random.random()),
                '_': str(ts),
            }

            try:
                resp = requests.get(base_url, params=params, headers=headers, timeout=15)
                data = parse_jsonp(resp.text)

                if data is None:
                    consecutive_empty += 1
                    if consecutive_empty >= 3:
                        print(f"    连续{consecutive_empty}次无数据，停止")
                        break
                    continue

                records = extract_records(data)
                if not records:
                    consecutive_empty += 1
                    if consecutive_empty >= 3:
                        break
                    continue

                consecutive_empty = 0
                existing = {r['期号'] for r in all_data}
                new_recs = [r for r in records if r['期号'] not in existing]
                all_data.extend(new_recs)

                print(f"    第{page}页: +{len(new_recs)} 条 (累计 {len(all_data)})")
                time.sleep(0.3)

            except Exception as e:
                print(f"    第{page}页失败: {e}")
                consecutive_empty += 1
                if consecutive_empty >= 3:
                    break

    # ================================================================
    # 策略3：其他 transactionType
    # ================================================================
    if not all_data:
        print("\n[3] 尝试其他 transactionType...")
        for tt in ['10001002', '10001003', '10001104', '10001105',
                    '10001001', '10001008', '10001007']:
            ts = int(time.time() * 1000)
            params = {
                'callback': f'jQuery_{ts}',
                'transactionType': tt,
                'lotteryId': '287',
                'issueCount': '1000',
                'pageNum': '1',
                'pageSize': '1000',
                'tt': str(random.random()),
                '_': str(ts),
            }
            try:
                resp = requests.get(base_url, params=params, headers=headers, timeout=10)
                text = resp.text.strip()
                if len(text) < 50:
                    continue

                data = parse_jsonp(text)
                if data is None:
                    continue

                data_str = json.dumps(data, ensure_ascii=False)
                if len(data_str) > 200:
                    print(f"    TT={tt}: {len(text)}字节, 预览: {data_str[:300]}")

                records = extract_records(data)
                if records and len(records) > len(all_data):
                    all_data = records
                    print(f"    ✓ TT={tt} 解析出 {len(records)} 条!")

            except Exception:
                continue

    # ================================================================
    # 保存
    # ================================================================
    if all_data:
        seen = set()
        unique = []
        for row in all_data:
            if row['期号'] not in seen:
                seen.add(row['期号'])
                unique.append(row)
        unique.sort(key=lambda x: x['期号'], reverse=True)

        # 最终验证
        print(f"\n{'─' * 70}")
        validate_records(unique)

        save_to_csv(unique)
    else:
        print("\n⚠ 未获取到数据")
        print("请打开 raw_api_response.json 查看返回的数据结构")


def parse_jsonp(text):
    """解析JSONP响应"""
    text = text.strip()
    m = re.match(r'[^(]+\(\s*([\s\S]*)\s*\)\s*;?\s*$', text)
    if m:
        text = m.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def extract_records(data):
    """从API响应中提取开奖记录"""
    if not isinstance(data, dict):
        return []

    records = []

    # ---- 查找包含记录的列表 ----
    record_list = None
    for key in ['data', 'result', 'list', 'rows', 'datas', 'items',
                 'records', 'content', 'lotteryResults', 'kjsjjh',
                 'dataList', 'historyList']:
        if key in data:
            val = data[key]
            if isinstance(val, list) and val and isinstance(val[0], dict):
                record_list = val
                break
            elif isinstance(val, dict):
                for k2 in ['list', 'data', 'rows', 'records']:
                    if k2 in val and isinstance(val[k2], list):
                        record_list = val[k2]
                        break
                if record_list:
                    break

    if not record_list:
        for k, v in data.items():
            if isinstance(v, list) and len(v) > 3 and v and isinstance(v[0], dict):
                record_list = v
                break

    if record_list:
        for item in record_list:
            row = parse_record(item)
            if row:
                records.append(row)
        return records

    # ---- 平行数组格式 ----
    issues = data.get('issue', [])
    if isinstance(issues, list) and issues:
        for k, v in data.items():
            if k == 'issue' or not isinstance(v, list) or len(v) != len(issues):
                continue
            sample = str(v[0]) if v else ''
            # ★ 修复：用 \d+ 匹配完整数字
            nums_in_sample = re.findall(r'\d+', sample)
            if len(nums_in_sample) >= 7:
                dates = []
                for dk in ['date', 'openTime', 'drawDate', 'openDate']:
                    if dk in data and isinstance(data[dk], list) and len(data[dk]) == len(issues):
                        dates = data[dk]
                        break

                for i in range(len(issues)):
                    code = str(v[i])
                    parsed = parse_lottery_numbers(code)
                    if parsed and len(parsed) == 7:
                        records.append({
                            '期号': str(issues[i]),
                            '开奖日期': str(dates[i])[:10] if i < len(dates) else '',
                            '号码1': parsed[0], '号码2': parsed[1], '号码3': parsed[2],
                            '号码4': parsed[3], '号码5': parsed[4], '号码6': parsed[5],
                            '号码7': parsed[6],
                        })
                return records

    return records


def parse_record(item):
    """解析单条开奖记录 — 全面支持第7位两位数"""
    if not isinstance(item, dict):
        return None

    row = {}

    # === 期号 ===
    for k in ['qh', 'qi', 'issue', 'issueNo', 'expect', 'lotteryDrawNum',
              'drawIssue', 'termNum', 'issueName', 'periodId', 'periodName']:
        if k in item and item[k]:
            row['期号'] = str(item[k]).strip()
            break
    if '期号' not in row:
        return None

    # === 日期 ===
    for k in ['kjsjA', 'kjsj', 'drawDate', 'openTime', 'date', 'openDate',
              'bonusDate', 'lotteryDrawTime', 'drawTime', 'kjDate']:
        if k in item and item[k]:
            dm = re.search(r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})', str(item[k]))
            row['开奖日期'] = dm.group(1) if dm else str(item[k])[:10]
            break
    row.setdefault('开奖日期', '')

    # === 号码（★ 核心修复区域） ===
    numbers = None

    # ---- 1) 前区+后区数组 ----
    for qq_key, hq_key in [('kjjgqqW', 'kjjghqW'), ('frontNums', 'backNums')]:
        if qq_key in item and hq_key in item:
            numbers = parse_number_arrays(item[qq_key], item[hq_key])
            if numbers:
                break

    # ---- 2) 前区+后区字符串 ----
    if not numbers:
        for fk, bk in [('frontWinningNum', 'backWinningNum'),
                        ('kjjgqq', 'kjjghq'),
                        ('redCode', 'blueCode')]:
            if fk in item and item[fk]:
                numbers = parse_front_back(item[fk], item.get(bk, ''))
                if numbers:
                    break

    # ---- 3) 完整号码字符串 ----
    if not numbers:
        for k in ['openCode', 'drawResult', 'code', 'number', 'result',
                   'kjjg', 'drawCode', 'winNumber', 'bonusCode',
                   'lotteryDrawResult', 'kjNum']:
            if k in item and item[k]:
                numbers = parse_lottery_numbers(item[k])
                if numbers:
                    break

    # ---- 4) 最后尝试：遍历所有字符串值 ----
    if not numbers:
        for k, v in item.items():
            if isinstance(v, str) and len(v) >= 7:
                numbers = parse_lottery_numbers(v)
                if numbers:
                    break

    if numbers and len(numbers) == 7:
        for i in range(7):
            row[f'号码{i + 1}'] = str(numbers[i])
        return row

    return None


# ============================================================
#  数据验证
# ============================================================
def validate_records(records):
    """验证并报告数据质量"""
    total = len(records)
    if total == 0:
        print("  ⚠ 无数据可验证")
        return

    pos7_values = []
    errors = []

    for rec in records:
        # 检查位置1~6: 必须是0~9
        for i in range(1, 7):
            val = int(rec.get(f'号码{i}', -1))
            if not (0 <= val <= 9):
                errors.append(f"  期{rec['期号']} 位置{i}={val} 超出0~9")

        # 检查位置7: 必须是0~14
        val7 = int(rec.get('号码7', -1))
        pos7_values.append(val7)
        if not (0 <= val7 <= 14):
            errors.append(f"  期{rec['期号']} 位置7={val7} 超出0~14")

    # 统计位置7的分布
    from collections import Counter
    p7_counter = Counter(pos7_values)

    print(f"\n  📊 数据验证报告 ({total} 条)")
    print(f"  {'─' * 60}")

    # 位置7分布
    print(f"  位置7 数值分布 (范围0~14):")
    print(f"  ", end="")
    for v in range(15):
        count = p7_counter.get(v, 0)
        pct = count / total * 100
        print(f"  {v:>2}:{count:>3}({pct:>4.1f}%)", end="")
        if v == 7:
            print()
            print(f"  ", end="")
    print()

    # 两位数统计
    two_digit_count = sum(1 for v in pos7_values if v >= 10)
    print(f"\n  位置7 两位数(10~14): {two_digit_count} 条 ({two_digit_count/total*100:.1f}%)")
    print(f"  位置7 单位数(0~9):  {total - two_digit_count} 条 ({(total-two_digit_count)/total*100:.1f}%)")

    if errors:
        print(f"\n  ⚠ 发现 {len(errors)} 个异常:")
        for e in errors[:10]:
            print(f"    {e}")
        if len(errors) > 10:
            print(f"    ... 还有 {len(errors) - 10} 个")
    else:
        print(f"\n  ✅ 全部 {total} 条数据验证通过!")


# ============================================================
#  保存
# ============================================================
def save_to_csv(data):
    filename = '7星彩_历史开奖号码.csv'
    fieldnames = ['期号', '开奖日期', '号码1', '号码2', '号码3', '号码4', '号码5', '号码6', '号码7']

    with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow({k: row.get(k, '') for k in fieldnames})

    print(f"\n{'=' * 70}")
    print(f"  ✓ 数据保存成功!")
    print(f"  文件名:   {filename}")
    print(f"  总记录:   {len(data)} 条")
    if data:
        n1 = ' '.join([data[0].get(f'号码{i}', '') for i in range(1, 8)])
        n2 = ' '.join([data[-1].get(f'号码{i}', '') for i in range(1, 8)])
        print(f"  最新一期: {data[0]['期号']}  {data[0].get('开奖日期','')}  [{n1}]")
        print(f"  最早一期: {data[-1]['期号']}  {data[-1].get('开奖日期','')}  [{n2}]")

        # 展示几条包含两位数的记录
        two_digit_samples = [r for r in data if int(r.get('号码7', 0)) >= 10][:5]
        if two_digit_samples:
            print(f"\n  两位数(位置7≥10)示例:")
            for r in two_digit_samples:
                nums = ' '.join([r.get(f'号码{i}', '') for i in range(1, 8)])
                print(f"    {r['期号']}  {r.get('开奖日期','')}  [{nums}]")

    print(f"{'=' * 70}")

    # 精简版
    sf = '7星彩_号码序列.csv'
    with open(sf, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['期号', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7'])
        for row in data:
            w.writerow([row['期号']] + [row.get(f'号码{i}', '') for i in range(1, 8)])
    print(f"  ✓ 精简版: {sf}")


if __name__ == '__main__':
    scrape_7star()
