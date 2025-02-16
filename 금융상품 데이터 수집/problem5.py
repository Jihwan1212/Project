import requests
import pprint

def get_savings_products():
    """
    금융상품통합비교공시 API를 사용하여 적금 상품 정보를 가져옵니다.
    """
    api_key = '7be1bc61be7a7f60ee0d2d9a764792a0'
    url = f'http://finlife.fss.or.kr/finlifeapi/savingProductsSearch.json?auth={api_key}&topFinGrpNo=020000&pageNo=1'

    # API 요청 및 JSON 데이터 가져오기
    response = requests.get(url).json()

    # baseList: 적금 상품 기본 정보
    savings_products = response.get("result", {}).get("baseList", [])
    
    # optionList: 적금 상품 옵션 정보 (금리 포함)
    savings_options = response.get("result", {}).get("optionList", [])

    return savings_products, savings_options

def get_highest_interest_savings():
    """
    API 데이터를 기반으로 금리가 가장 높은 적금 상품을 반환합니다.
    """
    savings_products, savings_options = get_savings_products()

    # 금리가 가장 높은 상품 찾기
    highest_interest_product = None
    highest_interest_rate = 0.0

    for option in savings_options:
        # 최고 우대금리 (intr_rate2)를 기준으로 비교
        interest_rate = option.get("intr_rate2", 0)
        
        if interest_rate > highest_interest_rate:
            highest_interest_rate = interest_rate
            highest_interest_product = option

    # 금리가 가장 높은 상품이 있을 경우, 상세 정보 추출
    if highest_interest_product:
        fin_prdt_cd = highest_interest_product["fin_prdt_cd"]

        # 해당 금융상품 코드와 일치하는 기본 정보 찾기
        product_info = next((p for p in savings_products if p["fin_prdt_cd"] == fin_prdt_cd), {})

        # 최종 결과 딕셔너리 생성
        result = {
            "금융회사명": product_info.get("kor_co_nm", ""),
            "금융상품명": product_info.get("fin_prdt_nm", ""),
            "저축 기간": highest_interest_product.get("save_trm", ""),
            "저축 금리": highest_interest_product.get("intr_rate", 0),
            "최고 우대금리": highest_interest_rate
        }
        
        return result

    return None  # 적금 상품이 없을 경우

# 실행 및 결과 출력
if __name__ == '__main__':
    highest_savings = get_highest_interest_savings()
    pprint.pprint(highest_savings)  # JSON 데이터를 보기 좋은 형식으로 출력

##########################################################################################
# Chat GPT 질문 프롬프트
##########################################################################################
# 다음의 문제를 해결할 수 있는 코드를 작성해주세요.
# 문제:
# 금융상품통합비교공시 API 를 이용하여 금리가 가장 높은 적금 상품을 가져오세요.
##########################################################################################