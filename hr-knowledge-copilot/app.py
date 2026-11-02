"""HR Knowledge Copilot Streamlit interface."""
from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from hr_copilot.config import ConfigError, load_settings
from hr_copilot.ingest import ingest_directory
from hr_copilot.llm import LLMError
from hr_copilot.qa import answer_question
from hr_copilot.retrieve import search_chunks

DATA_DIR = ROOT / "data" / "demo"
EVAL_FILE = ROOT / "rag_eval_results.json"
st.set_page_config(page_title="知序 · HR知识助手", page_icon="✦", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Noto+Sans+SC:wght@400;500;600;700&display=swap');
:root{--navy:#111c30;--blue:#3978f6;--ink:#1b2940;--muted:#687991;--line:#e8edf4;--canvas:#f3f6fb;--text-color:#1b2940;--primary-color:#3978f6}
html,body,[class*="css"]{font-family:'DM Sans','Noto Sans SC',sans-serif;color:var(--ink)}
.stApp,[data-testid="stAppViewContainer"]{color:var(--ink)!important;--text-color:var(--ink)}
.st-key-profile_details_card [data-testid="stTextInput"] [data-baseweb="input"],
.st-key-profile_details_card [data-testid="stTextInput"] [data-baseweb="input"]>div,
.st-key-profile_details_card [data-testid="stTextInput"] input{background:#fff!important;background-color:#fff!important;color:#111c30!important;-webkit-text-fill-color:#111c30!important;caret-color:#111c30!important}
.st-key-profile_details_card [data-testid="stWidgetLabel"],.st-key-profile_details_card [data-testid="stWidgetLabel"] *{color:#1b2940!important;-webkit-text-fill-color:#1b2940!important}
section.main,section.main p,section.main li,section.main label,
section.main h1,section.main h2,section.main h3,section.main h4,
section.main [data-testid="stMarkdownContainer"],section.main [data-testid="stWidgetLabel"],
section.main [data-testid="stCaptionContainer"],section.main [data-testid="stAlert"]{color:var(--ink)}
section.main input,section.main textarea,section.main [role="combobox"],
section.main [data-baseweb="select"] *{color:var(--ink)!important;-webkit-text-fill-color:var(--ink)}
section.main input::placeholder,section.main textarea::placeholder{color:#687991!important;-webkit-text-fill-color:#687991;opacity:1}
section.main div.stButton>button{color:#283751!important;-webkit-text-fill-color:#283751}
section.main div.stButton>button[kind="primary"]{color:#fff!important;-webkit-text-fill-color:#fff}
section.main [data-testid="stAlert"] *{color:var(--ink)!important;-webkit-text-fill-color:var(--ink)}
[data-testid="stDownloadButton"] button,
[data-testid="stDownloadButton"] button[kind="secondary"],
.stDownloadButton button{background:#3978f6!important;background-color:#3978f6!important;border:1px solid #3978f6!important;border-radius:10px!important;color:#fff!important;-webkit-text-fill-color:#fff!important;padding:10px 16px!important;min-height:42px!important;font-weight:600!important;box-shadow:none!important}
[data-testid="stDownloadButton"] button *,
.stDownloadButton button *{color:#fff!important;-webkit-text-fill-color:#fff!important;fill:#fff!important}
[data-testid="stDownloadButton"] button:hover,
.stDownloadButton button:hover{background:#2f68dc!important;background-color:#2f68dc!important;border-color:#2f68dc!important}
[data-testid="stDownloadButton"] button:focus-visible,
.stDownloadButton button:focus-visible{outline:3px solid rgba(57,120,246,.35)!important;outline-offset:2px!important}
[data-testid="stAppDeployButton"]{display:none!important}
#MainMenu{visibility:hidden!important}
.stApp{background:radial-gradient(circle at 12% 17%,rgba(57,120,246,.07) 0 1px,transparent 2px),radial-gradient(circle at 76% 12%,rgba(57,120,246,.06) 0 1px,transparent 2px),radial-gradient(circle at 48% 42%,rgba(57,120,246,.045) 0 1px,transparent 2px),#f2f6fc;background-size:110px 110px,150px 150px,190px 190px,auto} header[data-testid="stHeader"]{background:var(--navy)}
[data-testid="stSidebar"] {
    background: var(--navy);
    border-right: 0;
    z-index: 998;
}

[data-testid="stSidebar"] * {
    color: #dce5f4;
}
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapsedControl"] button{display:flex!important;visibility:visible!important;opacity:1!important;pointer-events:auto!important;z-index:1101!important;color:#fff!important}
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapsedControl"] button{background:#1c2a40!important;border:1px solid #64748b!important;border-radius:8px!important;min-width:34px!important;min-height:34px!important}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{color:#97a8c1}
section.main>div{padding:86px 2rem 64px;max-width:1600px}
header[data-testid="stHeader"] {
    background: var(--navy);
}
.topbar{position:fixed;top:0;left:0;width:100vw;height:74px;z-index:999;pointer-events:none;background:var(--navy);color:white;border-radius:0;padding:0 24px;margin:0;display:flex;align-items:center;justify-content:space-between;box-shadow:0 8px 24px rgba(10,20,38,.16);box-sizing:border-box}
.hero{text-align:center;margin:4px 0 12px}.hero h1{font-size:32px!important;margin:5px 0}.hero p{margin-bottom:10px}
.brand{position:absolute;left:24px;top:50%;transform:translateY(-50%);z-index:1002;pointer-events:none;display:flex;align-items:center;color:#fff;font-size:27px;font-weight:700;letter-spacing:.3px;white-space:nowrap;line-height:1.2}.brand .brand-mark{display:inline-block;color:#fff;font-size:30px;line-height:1;margin-right:5px}.brand small{display:inline-block;font-size:13px;color:#aab6c8;margin-left:10px;font-weight:400;line-height:1.2}
#top-profile-icon{position:fixed;top:18px;right:30px;z-index:10000;width:40px;height:40px;display:flex;align-items:center;justify-content:center;cursor:pointer;color:#aab6c8;text-decoration:none;pointer-events:auto;transition:color .2s}
#top-profile-icon:hover{color:#fff}
#top-profile-icon svg{width:24px;height:24px;display:block;fill:none;stroke:currentColor;stroke-width:1.7;stroke-linecap:round;stroke-linejoin:round}
.st-key-profile_logout div.stButton>button{color:#bd3f47!important;border-color:#e9b8bc!important;background:#fff!important}
.st-key-profile_logout div.stButton>button:hover{background:#fff5f5!important;border-color:#bd3f47!important}
.eyebrow{font-size:12px;font-weight:700;color:var(--blue);letter-spacing:1.2px;text-transform:uppercase}
.hero h1{font-size:30px;margin:8px 0 4px;color:#18263c}.hero p{color:var(--muted);margin:0 0 18px}
.card{background:#fff;border:1px solid #e8edf4;border-radius:16px;padding:22px 24px;box-shadow:0 8px 24px rgba(27,49,82,.045);margin-bottom:16px}
.st-key-faq_card{background:#fff;border:1px solid #e8edf4;border-radius:16px;padding:18px 20px;box-shadow:0 8px 24px rgba(27,49,82,.045);margin-bottom:16px}
.st-key-faq_card div.stButton:has(button[kind="secondary"]) button{background:transparent!important;border:0!important;box-shadow:none!important;min-height:28px!important;height:28px!important;padding:0!important;color:#3978f6!important;text-align:right!important}
.card-title{font-size:16px;font-weight:700;margin-bottom:14px}.pill{background:#edf4ff;color:#3978f6;padding:5px 10px;border-radius:20px;font-size:12px;font-weight:600}
.source{padding:12px 0;border-top:1px solid #edf0f5;display:flex;justify-content:space-between;align-items:center}.source:first-of-type{border-top:0}
.muted{color:#8290a5;font-size:13px}.faq{padding:11px 0;border-bottom:1px solid #edf0f5;font-size:14px}.faq:last-child{border:0}
.metric{font-size:27px;font-weight:700;color:#1b2940}.metric-label{color:#8290a5;font-size:12px}
div.stButton>button{border-radius:10px;border:1px solid #e4eaf2;background:white;color:#283751;text-align:left;min-height:43px}
div.stButton>button:hover{border-color:#3978f6;color:#3978f6}
div.stButton>button[kind="primary"]{background:#3978f6!important;border-color:#3978f6!important;color:#fff!important}
div[data-testid="stForm"]{background:white;padding:20px;border:1px solid #e8edf4;border-radius:16px;box-shadow:0 8px 24px rgba(27,49,82,.045)}
div.stFormSubmitButton>button{border-radius:10px}
.stTextArea textarea{border:0!important;background:transparent!important;font-size:16px!important;box-shadow:none!important}
.source-icon{display:inline-flex;width:29px;height:34px;border-radius:7px;background:#fff0ef;color:#e45d54;align-items:center;justify-content:center;font-size:10px;font-weight:700;margin-right:10px}
.answer-note{padding:12px 14px;border-radius:10px;background:#eef5ff;border-left:3px solid #3978f6;color:#385276;margin:12px 0}
[data-testid="stSidebar"]{padding-top:0}
[data-testid="stSidebar"] div.stButton>button[kind="primary"]{background:#3978f6!important;color:#fff!important;border-radius:12px!important}
.compact-faq div.stButton>button{border:0!important;background:transparent!important;min-height:30px!important;padding:4px 0!important;font-size:13px!important}
.compact-faq div.stButton>button:hover{color:#3978f6!important}
.assistant-stage{height:245px;display:flex;align-items:center;justify-content:center;flex-direction:column;text-align:center;color:#8595ad}
.compact-faq div.stButton>button{height:38px;min-height:38px!important;text-align:left!important;padding:0 14px!important;white-space:nowrap;display:flex!important;justify-content:flex-start!important;align-items:center!important;border-radius:10px!important;margin:0!important}
.compact-faq div.stButton>button>div{width:100%;display:flex!important;justify-content:flex-start!important;align-items:center!important}
.compact-faq div.stButton>button p{width:100%;text-align:left!important;margin:0!important;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.stChatInputContainer{position:fixed!important;bottom:10px!important;padding:0 2rem!important;background:transparent!important;z-index:100}
.stChatInputContainer>div{max-width:calc(1600px - 4rem);margin:auto;background:#fff;border:1px solid #e3eaf4;border-radius:16px;box-shadow:0 8px 24px rgba(27,49,82,.09)}
@media(max-height:760px){section.main>div{padding-top:78px}.assistant-stage{height:205px}.bot-orb{width:94px;height:94px;font-size:40px}.hero{margin:0 0 8px}.card{padding:16px 20px;margin-bottom:10px}}
.bot-orb{width:116px;height:116px;border-radius:38px;background:linear-gradient(145deg,#fff,#e7f0ff);box-shadow:0 18px 40px rgba(57,120,246,.16),inset 0 0 0 1px #dce8fb;display:grid;place-items:center;font-size:48px;animation:floaty 3.4s ease-in-out infinite;position:relative}
.bot-orb:after{content:'';position:absolute;inset:-11px;border:1px solid rgba(57,120,246,.15);border-radius:44px;animation:pulse 2.8s ease-out infinite}
@keyframes floaty{0%,100%{transform:translateY(0)}50%{transform:translateY(-9px)}}@keyframes pulse{0%{transform:scale(.94);opacity:.8}100%{transform:scale(1.12);opacity:0}}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner="正在读取知识库…")
def load_corpus():
    return ingest_directory(DATA_DIR)

chunks, warnings = load_corpus()

@st.cache_resource
def simulated_user_store():
    """Process-local stand-in for a persistent user database."""
    return {}

if "current_step" not in st.session_state:
    st.session_state.current_step = "login"

if st.query_params.get("nav") == "profile":
    del st.query_params["nav"]
    if st.session_state.current_step not in {"login", "onboarding"}:
        st.session_state.current_step = "profile"
        st.session_state.current_page = "个人空间"
        st.session_state.page = "个人空间"

def apply_user_profile(profile):
    st.session_state.profile_name = profile.get("name", "")
    st.session_state.profile_department = profile.get("department", "")
    st.session_state.profile_position = profile.get("position", "")

def render_auth_styles():
    st.markdown("""
    <style>
    [data-testid="stSidebar"]{display:none!important}
    .st-key-login_card,.st-key-onboarding_card{max-width:400px;width:calc(100% - 32px);margin:5vh auto 0;padding:30px 30px 28px;background:#fff;border:1px solid #e5ebf3;border-radius:20px;box-shadow:0 20px 60px rgba(25,48,82,.12);box-sizing:border-box}
    .auth-brand{text-align:center;color:#17253b;font-size:30px;font-weight:700;margin-bottom:6px}
    .auth-brand span{color:#3978f6;font-size:32px;margin-right:7px}
    .auth-caption{text-align:center;color:#8290a5;margin-bottom:22px}
    .st-key-login_card input,.st-key-onboarding_card input,[data-testid="stForm"] input{background:#fff!important;background-color:#fff!important;color:#111c30!important;-webkit-text-fill-color:#111c30!important;caret-color:#111c30!important;border-radius:10px!important;padding:12px 14px!important;min-height:44px}
    .st-key-login_card input::placeholder,.st-key-onboarding_card input::placeholder,[data-testid="stForm"] input::placeholder{color:#687991!important;-webkit-text-fill-color:#687991!important;opacity:1}
    .st-key-login_card input:-webkit-autofill,.st-key-onboarding_card input:-webkit-autofill,[data-testid="stForm"] input:-webkit-autofill{-webkit-text-fill-color:#111c30!important;box-shadow:0 0 0 1000px #fff inset!important;caret-color:#111c30!important}
    .st-key-login_card [data-testid="stWidgetLabel"] p,.st-key-onboarding_card [data-testid="stWidgetLabel"] p{font-weight:600;color:#34445d}
    .st-key-login_card div.stButton>button[kind="primary"],.st-key-onboarding_card div.stFormSubmitButton>button{width:100%;min-height:46px;border-radius:11px;background:#3978f6!important;border-color:#3978f6!important;color:#fff!important}
    .st-key-login_card div.stButton>button:not([kind="primary"]){background:#F5F7FA;border:0;min-height:44px;border-radius:10px;color:#3978f6}
    .st-key-onboarding_card [data-testid="stForm"]{border:0;box-shadow:none;padding:0;background:transparent}
    div.stButton>button[kind="primary"]{background:#3978f6!important;border-color:#3978f6!important;color:white!important}
    </style>
    """,unsafe_allow_html=True)

if st.session_state.current_step == "login":
    render_auth_styles()
    @st.fragment(run_every="1s")
    def login_screen():
        with st.container(key="login_card"):
            st.markdown("<div class='auth-brand'><span>✦</span>知序</div><div class='auth-caption'>HR 知识助手 · 安全登录</div>",unsafe_allow_html=True)
            phone=st.text_input("手机号",placeholder="请输入手机号",key="login_phone")
            sent_at=st.session_state.get("otp_sent_at",0.0)
            remaining=max(0,int(60-(time.time()-sent_at))) if sent_at else 0
            code_col,send_col=st.columns([2.1,1],gap="small",vertical_alignment="bottom")
            with code_col:
                code=st.text_input("验证码",placeholder="请输入验证码",key="login_code")
            with send_col:
                if st.button("重新获取" if remaining else "获取验证码",disabled=remaining>0,use_container_width=True,key="send_otp"):
                    st.session_state.otp_sent_at=time.time()
                    st.rerun()
            if remaining: st.caption(f"{remaining} 秒后可重发")
            if sent_at: st.caption("演示验证码：123456")
            if st.button("登录",type="primary",use_container_width=True,key="login_submit"):
                if not re.fullmatch(r"1\d{10}",phone.strip()): st.error("请输入有效的 11 位手机号。")
                elif code.strip()!="123456": st.error("验证码不正确，请输入演示验证码 123456。")
                else:
                    store=simulated_user_store()
                    st.session_state.phone=phone.strip()
                    if phone.strip() in store:
                        apply_user_profile(store[phone.strip()])
                        st.session_state.current_step="main"; st.session_state.page="智能问答"; st.session_state.current_page="智能问答"
                    else:
                        st.session_state.current_step="onboarding"
                    st.rerun()
    login_screen()
    st.stop()

if st.session_state.current_step == "onboarding":
    render_auth_styles()
    with st.container(key="onboarding_card"):
        st.markdown("<div class='auth-brand'><span>✦</span>知序</div><h2 style='text-align:center'>完善个人资料</h2><div class='auth-caption'>首次登录，请完善个人资料</div>",unsafe_allow_html=True)
        with st.form("onboarding_form"):
            new_name=st.text_input("姓名（必填）",placeholder="请输入姓名")
            new_department=st.selectbox("部门",["人力资源部","行政部","财务部","研发部","市场部","其他"])
            new_position=st.text_input("职位",placeholder="请输入职位")
            save=st.form_submit_button("保存并进入",type="primary",use_container_width=True)
        if save:
            if not new_name.strip(): st.error("请填写姓名后继续。")
            else:
                profile={"name":new_name.strip(),"department":new_department,"position":new_position.strip()}
                simulated_user_store()[st.session_state.phone]=profile
                apply_user_profile(profile)
                st.session_state.current_step="main"; st.session_state.page="智能问答"; st.session_state.current_page="智能问答"
                st.rerun()
    st.stop()

with st.sidebar:
    pages=["智能问答","知识库","历史记录","收藏夹","个人空间"]
    if "page" not in st.session_state: st.session_state.page=pages[0]
    if "current_page" not in st.session_state: st.session_state.current_page=st.session_state.page
    st.markdown("<style>[data-testid='stSidebar'] div.stButton>button{background:transparent;border:0;color:#b9c6d8;padding:9px 14px}[data-testid='stSidebar'] div.stButton>button:hover{background:#1d2d47;color:white}[data-testid='stSidebar'] div.stButton>button[kind='primary']{background:#3978f6;color:white}</style>",unsafe_allow_html=True)
    for page in pages[:5]:
        if st.button(("▰  " if st.session_state.current_page==page else "　 ")+page,key="nav_"+page,use_container_width=True,type="primary" if st.session_state.current_page==page else "secondary"):
            st.session_state.current_step="profile" if page=="个人空间" else "main"
            st.session_state.current_page=page
            st.session_state.page=page
            st.rerun()

st.markdown("""
<div class="topbar">
  <div class="brand"><span class="brand-mark">✦</span>知序 <small>HR 知识助手</small></div>
  <a id="top-profile-icon" href="?nav=profile" aria-label="打开个人空间" title="个人空间">
    <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="8" r="3.5"></circle><path d="M5.5 20c.7-3.5 3.1-5.5 6.5-5.5s5.8 2 6.5 5.5"></path></svg>
  </a>
</div>
""",unsafe_allow_html=True)
page=st.session_state.current_page
if st.session_state.current_step == "profile":
    st.markdown("<div class='hero'><div class='eyebrow'>PERSONAL SPACE</div><h1>个人空间</h1><p>查看并管理你的个人资料与账号信息。</p></div>",unsafe_allow_html=True)
    profile_left,profile_right=st.columns([1,1],gap="large")
    with profile_left:
        with st.container(key="profile_details_card",border=True):
            st.subheader("个人资料")
            name=st.text_input("姓名",value=st.session_state.get("profile_name","林同学"),key="profile_name_input")
            department=st.text_input("部门",value=st.session_state.get("profile_department","人力资源部"),key="profile_department_input")
            position=st.text_input("职位",value=st.session_state.get("profile_position","HR 专员"),key="profile_position_input")
            if st.button("保存资料",type="primary",key="save_profile"):
                st.session_state.profile_name=name; st.session_state.profile_department=department; st.session_state.profile_position=position
                phone=st.session_state.get("phone")
                if phone: simulated_user_store()[phone]={"name":name,"department":department,"position":position}
                st.success("个人资料已保存。")
    with profile_right:
        with st.container(border=True):
            st.subheader("账号设置")
            if st.button("修改密码",key="change_password"):
                st.info("此演示版本尚未接入账号认证服务，暂不能修改密码。")
            st.markdown("#### 安全记录")
            with st.expander("最近登录设备 / IP"):
                st.markdown("**设备：** 当前浏览器会话")
                st.markdown("**IP 地址：** 127.0.0.1（本地开发环境）")
                st.caption("当前演示环境不保存历史登录记录。")
            st.markdown("#### 消息通知")
            st.toggle("系统通知",value=st.session_state.get("system_notifications_enabled",True),key="system_notifications_enabled")
            st.toggle("业务通知",value=st.session_state.get("business_notifications_enabled",True),key="business_notifications_enabled")
    st.markdown("<div style='height:24px'></div><div style='border-top:1px solid #e4eaf2;padding-top:20px;color:#bd3f47;font-weight:700'>危险操作</div><div style='color:#8290a5;font-size:13px;margin:6px 0 12px'>退出当前会话并返回智能问答。</div>",unsafe_allow_html=True)
    if st.button("退出登录",key="profile_logout"):
        st.session_state.clear()
        st.rerun()
elif st.session_state.current_step=="main" and page=="智能问答":
    st.markdown("<div class='hero'><div class='eyebrow'>HR KNOWLEDGE ASSISTANT</div><h1>你好，今天想了解什么？</h1><p>基于企业制度知识库，为你提供准确、可追溯的解答。</p></div>",unsafe_allow_html=True)
    left,right=st.columns([1.75,1],gap="large")
    with left:
        if "last_answer" in st.session_state:
            q,answer,hits=st.session_state.last_answer
            # Source documents are presented by the retrieval panel; keep answer copy focused on the policy content.
            answer_lines=[line for line in answer.splitlines() if not line.strip().startswith(("来源：", "来源:"))]
            answer="\n".join(answer_lines).strip()
            st.markdown(f"<div class='card' style='background:#f9fbff;border-color:#e4ebf5'><div style='color:#7d8ba0;font-size:12px;margin-bottom:8px'>你　·　{q}</div></div>",unsafe_allow_html=True)
            st.markdown("<div class='card'><div class='card-title'>✦　知序为你解答 <span style='float:right;color:#8290a5;font-size:13px'>▢　复制　　♧　有帮助</span></div><div class='answer-note'>以下回答根据已检索到的企业制度整理，具体执行请以最新制度原文为准。</div>"+answer.replace("\n","<br>")+"</div>",unsafe_allow_html=True)
            favorites=st.session_state.setdefault("favorites",[])
            is_favorite=any(item.get("question")==q for item in favorites)
            if st.button("★　取消收藏" if is_favorite else "☆　收藏此回答",key="favorite_current_answer"):
                if is_favorite:
                    st.session_state.favorites=[item for item in favorites if item.get("question")!=q]
                else:
                    st.session_state.favorites=favorites+[{"question":q,"answer":answer}]
                st.rerun()
        else:
            st.markdown("<div class='assistant-stage'><div class='bot-orb'>✦</div><div style='font-size:17px;font-weight:600;color:#526784;margin-top:25px'>我是知序，你的 HR 制度小助手</div><div style='font-size:13px;margin-top:8px'>选择右侧常见问题，或在下方输入问题开始对话</div></div>",unsafe_allow_html=True)
    with right:
        with st.container(key="faq_card"):
            all_faqs=["年假有几天？","报销需要什么凭证？","出差住宿标准是多少？","事假期间发工资吗？","年假需要提前几天申请？","出差申请需要提前几天？"]
            faq_offset=st.session_state.get("faq_offset",0)
            title_col,shuffle_col=st.columns([3,1])
            title_col.markdown("<div class='card-title' style='margin:0;padding:5px 0 12px'>✨　常见问题</div>",unsafe_allow_html=True)
            if shuffle_col.button("换一换　↻",key="shuffle_faq"):
                st.session_state.faq_offset=(faq_offset+4)%len(all_faqs); st.rerun()
            visible_faqs=[all_faqs[(faq_offset+i)%len(all_faqs)] for i in range(4)]
            for faq in visible_faqs:
                dot_col,question_col,arrow_col=st.columns([0.18,3.4,0.25],gap="small")
                dot_col.markdown("<div style='color:#3978f6;font-size:13px;line-height:38px;text-align:center'>●</div>",unsafe_allow_html=True)
                if question_col.button(faq,key="faq_"+faq,use_container_width=True):
                    st.session_state.pending_question=faq; st.rerun()
                arrow_col.markdown("<div style='color:#93a0b3;font-size:16px;line-height:38px;text-align:right'>›</div>",unsafe_allow_html=True)
    submitted=st.chat_input("输入制度问题，按回车发送…")
    question=st.session_state.pop("pending_question",None) or submitted
    if question:
        hits=search_chunks(chunks,question).hits
        try:
            answer=answer_question(chunks,question,load_settings())
        except ConfigError:
            answer="当前未配置在线模型，以下为知识库检索到的制度原文，供你参考。" if hits else "知识库未找到足够的制度依据，无法回答该问题。"
        except LLMError:
            answer="在线问答暂不可用，请稍后重试。"
        st.session_state.last_answer=(question,answer,hits)
        st.session_state.history=st.session_state.get("history",[])+[(question,answer)]
        st.rerun()
elif st.session_state.current_step=="main" and page=="知识库":
    st.markdown("<div class='hero'><div class='eyebrow'>KNOWLEDGE BASE</div><h1>知识库</h1><p>浏览当前已接入的企业制度文档。</p></div>",unsafe_allow_html=True)
    docs=sorted(DATA_DIR.glob("*"))
    supported_docs=[p for p in docs if p.suffix.lower() in {".txt",".pdf"} and p.name!="DEMO_attendance_policy.pdf"]
    with st.container(border=True):
        st.markdown(f"### 📚 全部文档　<span class='pill'>{len(supported_docs)} 份</span>",unsafe_allow_html=True)
    for path in supported_docs:
        doc_chunks=[c for c in chunks if c.source_name==path.name]
        pages=len(set(c.page_number for c in doc_chunks))
        mime="application/pdf" if path.suffix.lower()==".pdf" else "text/plain"
        file_col,info_col,download_col=st.columns([5,3,1.5],vertical_alignment="center")
        file_col.markdown(f"**📄　{path.name}**")
        info_col.caption(f"{pages} 页　·　{len(doc_chunks)} 个片段　·　已索引")
        download_col.download_button("打开 / 下载",data=path.read_bytes(),file_name=path.name,mime=mime,key=f"download_{path.name}",use_container_width=True)
elif st.session_state.current_step=="main" and page=="质量评测":
    st.markdown("<div class='hero'><div class='eyebrow'>RAG EVALUATION</div><h1>质量评测</h1><p>查看最近一次 RAG 离线评测结果与失败案例。</p></div>",unsafe_allow_html=True)
    results=json.loads(EVAL_FILE.read_text(encoding="utf-8")) if EVAL_FILE.exists() else []
    done=[x for x in results if x.get("status")=="evaluated"]; total=max(len(done),1)
    metrics=[("答案命中率",sum(x.get("answer_ok",False) for x in done)),("来源准确率",sum(x.get("source_ok",False) for x in done)),("拒答准确率",sum(x.get("refusal_ok",False) for x in done))]
    cols=st.columns(3)
    for col,(label,value) in zip(cols,metrics): col.markdown(f"<div class='card'><div class='metric'>{value/total:.0%}</div><div class='metric-label'>{label}　{value}/{len(done)}</div></div>",unsafe_allow_html=True)
    failures=[x for x in done if not (x.get("answer_ok") and x.get("source_ok") and x.get("refusal_ok"))]
    st.markdown(f"<div class='card'><div class='card-title'>评测概览　<span class='pill'>{len(done)} 条样例</span></div><div class='muted'>基于本地 rag_eval_results.json 中最近一次评测记录。</div></div>",unsafe_allow_html=True)
    st.markdown(f"<div class='card'><div class='card-title'>失败案例　<span class='pill'>{len(failures)} 条</span></div>",unsafe_allow_html=True)
    if failures:
        for item in failures: st.markdown(f"<div class='source'><span>{item.get('question','')}</span><span class='pill'>需关注</span></div><div class='muted'>{item.get('answer','')}</div>",unsafe_allow_html=True)
    else: st.success("当前评测记录中暂无失败案例。")
    st.markdown("</div>",unsafe_allow_html=True)
elif st.session_state.current_step=="main":
    st.markdown(f"<div class='hero'><div class='eyebrow'>WORKSPACE</div><h1>{page}</h1><p>知序 · HR 知识助手</p></div>",unsafe_allow_html=True)
    if page=="历史记录":
        history=st.session_state.get("history",[])
        if not history: st.info("还没有问答记录。提交问题后，记录会显示在这里。")
        for q,a in reversed(history): st.markdown(f"<div class='card'><b>{q}</b><p>{a}</p></div>",unsafe_allow_html=True)
    elif page=="收藏夹":
        favorites=st.session_state.get("favorites",[])
        if not favorites:
            st.info("暂无收藏内容。回答问题后，可以点击“收藏此回答”保存到这里。")
        for index,item in enumerate(reversed(favorites)):
            with st.container(border=True):
                st.markdown(f"**{item['question']}**")
                st.write(item["answer"])
                if st.button("取消收藏",key=f"remove_favorite_{index}"):
                    st.session_state.favorites.remove(item)
                    st.rerun()
    else: st.markdown("<div class='card'><div class='card-title'>系统状态</div><div class='source'><span>本地知识文档</span><b>已连接</b></div><div class='source'><span>检索模块</span><b>关键词匹配</b></div><div class='source'><span>在线模型</span><b>按环境配置启用</b></div></div>",unsafe_allow_html=True)
