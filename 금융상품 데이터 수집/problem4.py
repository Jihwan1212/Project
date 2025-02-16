import pprint
import requests

# 상품과 옵션 정보들을 담고 있는 새로운 객체를 만들어 반환하시오.
# 상품 리스트와 옵션 리스트를 금융상품 코드를 기준으로 매칭할 수 있습니다.
# 아래와 같은 순서로 데이터를 출력하며 진행합니다.
# 1. 응답을 json 형식으로 변환합니다.
# 2. key 값이 "result" 인 데이터를 변수에 저장합니다.
# 3. 2번의 결과 중 key 값이 "baseList" 인 데이터를 변수에 저장합니다.
# 4. 2번의 결과 중 key 값이 "optionList" 인 데이터를 변수에 저장합니다.
# 5. 3번에서 저장된 변수를 순회하며, 4번에서 저장된 값들에서 금융 상품 코드가 
#     같은 모든 데이터들을 가져와 새로운 딕셔너리로 저장합니다.
#     저장 시, 명세서에 맞게 출력되도록 저장합니다.
# 6. 5번에서 만든 딕셔너리를 결과 리스트에 추가합니다.


def get_deposit_products():
    api_key = "7be1bc61be7a7f60ee0d2d9a764792a0"
    url = f'http://finlife.fss.or.kr/finlifeapi/depositProductsSearch.json?auth={api_key}&topFinGrpNo=020000&pageNo=1'
    
    # API 요청 및 JSON 데이터 가져오기
    response = requests.get(url).json()
    deposit_products = response.get("result", {}).get("baseList", [])
    deposit_options = response.get("result", {}).get("optionList", [])

    return deposit_products, deposit_options

def process_deposit_data():
    deposit_products, deposit_options = get_deposit_products()

    # 결과를 저장할 리스트
    refined_list = []

    # 금융상품 정보를 딕셔너리 형태로 저장
    product_dict = {}

    # 금융상품 기본 정보 처리
    for product in deposit_products:
        fin_prdt_cd = product["fin_prdt_cd"]
        
        # 새로운 딕셔너리 생성
        new_dict = {
            "금융회사명": product["kor_co_nm"],  # 금융회사명
            "금융상품명": product["fin_prdt_nm"],  # 금융상품명
            "금리정보": []  # 금리 정보 리스트 초기화
        }
        
        # 딕셔너리에 추가
        product_dict[fin_prdt_cd] = new_dict

    # 금융상품 옵션 정보 처리
    for option in deposit_options:
        fin_prdt_cd = option["fin_prdt_cd"]
        
        # 해당 금융상품 코드가 product_dict에 존재하는지 확인
        if fin_prdt_cd in product_dict:
            # 옵션 정보 딕셔너리 생성
            option_dict = {
                "저축 금리": option.get("intr_rate", 0),  # 기본 금리
                "저축 기간": option.get("save_trm", ""),  # 저축 기간
                "저축금리유형": option.get("intr_rate_type", ""),  # 금리 유형
                "저축금리유형명": option.get("intr_rate_type_nm", ""),  # 금리 유형명
                "최고 우대금리": option.get("intr_rate2", 0)  # 최고 우대금리
            }
            
            # 해당 금융상품의 "금리정보" 리스트에 옵션 추가
            product_dict[fin_prdt_cd]["금리정보"].append(option_dict)

    # 최종적으로 product_dict의 모든 딕셔너리를 리스트로 변환
    refined_list = list(product_dict.values())

    return refined_list

# 실행 및 결과 출력
if __name__ == '__main__':
    result = process_deposit_data()
    pprint.pprint(result)  # JSON 데이터를 보기 좋은 형식으로 출력