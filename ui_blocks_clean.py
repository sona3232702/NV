import streamlit as st
from typing import Dict, Any

def render_tv_prompt(prompt: Dict[str, Any]) -> None:
    ptype = prompt.get("type","TEXT")
    text = prompt.get("text","")

    if ptype == "COLOR":
        color = prompt.get("color","#777777")
        # Dark text overlay for readability
        st.markdown(f"""
        <div style="
            height:300px;
            border-radius:24px;
            background:{color};
            display:flex;
            align-items:center;
            justify-content:center;
            text-align:center;
            padding:24px;">
            <div style="font-size:44px;font-weight:900;color:rgba(0,0,0,0.65);line-height:1.1;">
                {text}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="
            height:300px;
            border-radius:24px;
            border:1px solid rgba(255,255,255,0.18);
            display:flex;
            align-items:center;
            justify-content:center;
            text-align:center;
            padding:24px;">
            <div style="font-size:44px;font-weight:900;line-height:1.1;">
                {text}
            </div>
        </div>
        """, unsafe_allow_html=True)
