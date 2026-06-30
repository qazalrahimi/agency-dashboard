import streamlit as st
import pandas as pd
import plotly.express as px
from groq import Groq
from dotenv import load_dotenv
import os

# ── Configuration ──────────────────────────────────────────────────────────────
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

st.set_page_config(
    page_title="Agency Operations Dashboard",
    page_icon="📊",
    layout="wide"
)

# ── RTL Styling ────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
        .stMarkdown, .stDataFrame, p, h1, h2, h3 { direction: rtl; text-align: right; }
        .stTextInput input { direction: rtl; text-align: right; }
    </style>
""", unsafe_allow_html=True)

# ── Helper Functions ───────────────────────────────────────────────────────────
def get_avatar_url(name: str) -> str:
    """Generate a consistent avatar based on a person's name."""
    return f"https://api.dicebear.com/7.x/initials/svg?seed={name}"

def get_brand_logo_url(brand: str) -> str:
    """Generate a consistent shape-based logo for a brand."""
    return f"https://api.dicebear.com/7.x/shapes/svg?seed={brand}"

def ask_ai(question: str, context: str) -> str:
    """Send analysis context + question to AI and return a Persian answer."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": f"""
You are an expert agency operations analyst.
You must respond ONLY in fluent, natural Persian (Farsi).
Do not use any English, Chinese, or Vietnamese words. Use proper Persian vocabulary only.

Data context:
{context}

Question: {question}

Give a clear, actionable answer in pure Persian.
            """
        }],
        max_tokens=800,
        temperature=0.3
    )
    return response.choices[0].message.content

@st.cache_data
def explode_team_members(df: pd.DataFrame) -> pd.DataFrame:
    """Split the Team_Members column so each person gets their own row."""
    df_people = df.copy()
    df_people['Team_Members'] = df_people['Team_Members'].str.split(', ')
    df_people = df_people.explode('Team_Members')
    return df_people.rename(columns={'Team_Members': 'Person'})

@st.cache_data
def compute_person_stats(df_people: pd.DataFrame) -> pd.DataFrame:
    """Aggregate workload and performance metrics per person."""
    stats = df_people.groupby('Person').agg(
        Total_Projects=('Project_Code', 'count'),
        Avg_Project_Duration=('Planned_Duration_Days', 'mean'),
        Avg_Delay=('Delay_Days', 'mean'),
        Completed_Rate=('Status', lambda x: (x == 'انجام‌شده').mean())
    ).round(2)
    return stats.sort_values('Total_Projects', ascending=False)

@st.cache_data
def compute_brand_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate workload, diversity and quality metrics per brand, plus Key Account Score."""
    stats = df.groupby('Brand').agg(
        Total_Projects=('Project_Code', 'count'),
        Units_Worked_With=('Unit', 'nunique'),
        Avg_Delay=('Delay_Days', 'mean'),
        Completed_Rate=('Status', lambda x: (x == 'انجام‌شده').mean()),
        Cancelled_Count=('Status', lambda x: (x == 'کنسل‌شده').sum())
    ).round(2)

    stats['Project_Score'] = stats['Total_Projects'] / stats['Total_Projects'].max()
    stats['Diversity_Score'] = stats['Units_Worked_With'] / stats['Units_Worked_With'].max()
    stats['Quality_Score'] = stats['Completed_Rate']
    stats['Key_Account_Score'] = (
        stats['Project_Score'] * 0.4 +
        stats['Diversity_Score'] * 0.3 +
        stats['Quality_Score'] * 0.3
    ).round(2)

    return stats.sort_values('Key_Account_Score', ascending=False)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📊 داشبورد عملکرد عملیات آژانس")
st.markdown("تحلیل پروژه‌ها، واحدها، افراد و برندها")
st.markdown("---")

# ── File Upload ────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("فایل CSV پروژه‌ها را آپلود کنید", type=["csv"])

if uploaded_file is None:
    st.info("برای شروع یک فایل CSV آپلود کنید")
    st.stop()

df = pd.read_csv(uploaded_file)
st.success(f"فایل بارگذاری شد: {len(df)} پروژه")

df_people = explode_team_members(df)
person_stats = compute_person_stats(df_people)
brand_stats = compute_brand_stats(df)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 آمار کلی", "📈 نمودارها", "🔍 تحلیل عمیق",
    "👤 عملکرد افراد", "🏢 عملکرد برندها", "🤖 سوال از AI"
])

# ── Tab 1: Overview ────────────────────────────────────────────────────────────
with tab1:
    st.subheader("آمار کلی")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("تعداد کل پروژه‌ها", len(df))
    col2.metric("تعداد برندها", df['Brand'].nunique())
    completed_rate = (df['Status'] == 'انجام‌شده').mean()
    col3.metric("نرخ تکمیل", f"{completed_rate:.0%}")
    avg_delay = df[df['Delay_Days'] > 0]['Delay_Days'].mean()
    col4.metric("میانگین تأخیر", f"{avg_delay:.1f} روز")

    col5, col6 = st.columns(2)

    with col5:
        status_counts = df['Status'].value_counts()
        fig = px.pie(
            values=status_counts.values, names=status_counts.index,
            title="توزیع وضعیت پروژه‌ها"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        unit_counts = df['Unit'].value_counts()
        fig = px.bar(
            x=unit_counts.index, y=unit_counts.values,
            title="تعداد پروژه به ازای واحد",
            labels={'x': 'واحد', 'y': 'تعداد پروژه'}
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("نمایش داده خام")
    st.dataframe(df.head(20), use_container_width=True)

# ── Tab 2: Charts ──────────────────────────────────────────────────────────────
with tab2:
    st.subheader("نمودارهای تحلیلی")

    col1, col2 = st.columns(2)

    with col1:
        delay_by_unit = df.groupby('Unit')['Delay_Days'].mean().sort_values(ascending=False)
        fig = px.bar(
            x=delay_by_unit.index, y=delay_by_unit.values,
            title="میانگین تأخیر به ازای واحد (روز)",
            labels={'x': 'واحد', 'y': 'میانگین تأخیر'},
            color=delay_by_unit.values, color_continuous_scale="Reds"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        revision_by_category = df.groupby('Category')['Revision_Count'].mean().sort_values(ascending=False)
        fig = px.bar(
            x=revision_by_category.index, y=revision_by_category.values,
            title="میانگین بازگشت به ازای دسته پروژه",
            labels={'x': 'دسته', 'y': 'میانگین بازگشت'},
            color=revision_by_category.values, color_continuous_scale="Oranges"
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        df, x='Revision_Count', y='Delay_Days', color='Unit',
        title="رابطه بین تعداد بازگشت و تأخیر",
        labels={'Revision_Count': 'تعداد بازگشت', 'Delay_Days': 'تأخیر (روز)'},
        opacity=0.6, trendline="ols"
    )
    st.plotly_chart(fig, use_container_width=True)

    month_counts = df['Month'].value_counts()
    fig = px.bar(
        x=month_counts.index, y=month_counts.values,
        title="تعداد پروژه به ازای ماه",
        labels={'x': 'ماه', 'y': 'تعداد پروژه'}
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 3: Deep Analysis ───────────────────────────────────────────────────────
with tab3:
    st.subheader("تحلیل عمیق")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**وضعیت پروژه‌ها به ازای واحد**")
        status_by_unit = pd.crosstab(df['Unit'], df['Status'])
        st.dataframe(status_by_unit, use_container_width=True)

    with col2:
        st.markdown("**دلایل هولد و کنسلی**")
        reason_counts = df[df['Status_Reason'] != '']['Status_Reason'].value_counts()
        fig = px.bar(
            x=reason_counts.values, y=reason_counts.index,
            orientation='h', title="علت‌های هولد و کنسلی پروژه‌ها"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**نرخ تایید نهایی مشتری**")
    approval_counts = df['Client_Approval_Status'].value_counts()
    fig = px.bar(
        x=approval_counts.index, y=approval_counts.values,
        title="وضعیت تایید مشتری",
        labels={'x': 'وضعیت تایید', 'y': 'تعداد پروژه'}
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 4: People Performance ──────────────────────────────────────────────────
with tab4:
    st.subheader("عملکرد افراد")

    top_n = st.slider("تعداد افراد نمایش داده‌شده", 5, len(person_stats), 10)
    top_people = person_stats.head(top_n)

    fig = px.bar(
        x=top_people['Total_Projects'], y=top_people.index,
        orientation='h', title="حجم کار افراد (تعداد پروژه)",
        labels={'x': 'تعداد پروژه', 'y': 'فرد'},
        color=top_people['Completed_Rate'], color_continuous_scale="Greens"
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**کارت‌های افراد برتر**")
    cols = st.columns(5)
    for i, (person, row) in enumerate(top_people.head(5).iterrows()):
        with cols[i % 5]:
            st.image(get_avatar_url(person), width=60)
            st.markdown(f"**{person}**")
            st.caption(f"{row['Total_Projects']} پروژه")
            st.caption(f"نرخ تکمیل: {row['Completed_Rate']:.0%}")

    st.markdown("**جدول کامل عملکرد افراد**")
    st.dataframe(person_stats, use_container_width=True)

# ── Tab 5: Brand Performance ────────────────────────────────────────────────────
with tab5:
    st.subheader("عملکرد برندها — شناسایی مشتریان کلیدی")

    top_brands_n = st.slider("تعداد برندهای نمایش داده‌شده", 5, len(brand_stats), 10)
    top_brands = brand_stats.head(top_brands_n)

    fig = px.bar(
        x=top_brands.index, y=top_brands['Total_Projects'],
        title="حجم کار برندها",
        labels={'x': 'برند', 'y': 'تعداد پروژه'},
        color=top_brands['Completed_Rate'], color_continuous_scale="Greens"
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**برندهای برتر — امتیاز حساب کلیدی**")
    cols = st.columns(5)
    for i, (brand, row) in enumerate(top_brands.head(5).iterrows()):
        with cols[i % 5]:
            st.image(get_brand_logo_url(brand), width=60)
            st.markdown(f"**{brand}**")
            st.caption(f"امتیاز: {row['Key_Account_Score']}")
            st.caption(f"{row['Total_Projects']} پروژه")

    st.markdown("**جدول کامل امتیاز برندها**")
    st.dataframe(
        brand_stats[['Total_Projects', 'Units_Worked_With', 'Completed_Rate', 'Cancelled_Count', 'Key_Account_Score']],
        use_container_width=True
    )

# ── Tab 6: AI Chat ─────────────────────────────────────────────────────────────
with tab6:
    st.subheader("سوال از AI")

    delay_by_unit = df.groupby('Unit')['Delay_Days'].mean()
    context = f"""
Key Account Scores (top brands):
{brand_stats[['Total_Projects', 'Completed_Rate', 'Key_Account_Score']].head(10).to_string()}

Unit Delay Analysis:
{delay_by_unit.to_string()}

Top performers:
{person_stats[['Total_Projects', 'Completed_Rate']].head(10).to_string()}
"""

    st.markdown("**سوالات پیشنهادی:**")
    col1, col2, col3 = st.columns(3)

    suggested_questions = {
        "col1": "کدوم واحد بیشترین تأخیر رو داره و چرا؟",
        "col2": "مهم‌ترین مشتریان ما کدوم برندها هستن؟",
        "col3": "چطور نرخ بازگشت پروژه‌ها رو کم کنیم؟"
    }

    if col1.button(suggested_questions["col1"]):
        with st.spinner("در حال تحلیل..."):
            st.markdown(ask_ai(suggested_questions["col1"], context))

    if col2.button(suggested_questions["col2"]):
        with st.spinner("در حال تحلیل..."):
            st.markdown(ask_ai(suggested_questions["col2"], context))

    if col3.button(suggested_questions["col3"]):
        with st.spinner("در حال تحلیل..."):
            st.markdown(ask_ai(suggested_questions["col3"], context))

    st.markdown("---")

    user_question = st.text_input("سوال خودت رو بپرس:")
    if st.button("تحلیل کن") and user_question:
        with st.spinner("در حال تحلیل..."):
            st.markdown(ask_ai(user_question, context))
