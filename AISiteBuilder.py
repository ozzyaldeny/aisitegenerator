import streamlit as st
from streamlit_ace import st_ace, KEYBINDINGS, LANGUAGES, THEMES

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI,OpenAIChat
from langchain.chains import LLMChain, SimpleSequentialChain

import openai



if ('frontend_finished' not in st.session_state ):
    st.session_state['frontend_finished'] = False

def clear_text():
    st.session_state["refine_text"]=""


def generate_website_langchain(text):
    llm = OpenAIChat(temperature=0.0,model="gpt-3.5-turbo",max_tokens=3900, openai_api_key=st.session_state['OpenAPIKey'])
    prompt = PromptTemplate(
    input_variables=["text"],
    template="""
       You will act as frontend web developer and output necessary html, css, javascript code. User will describe what the web page will contain in delimeter <text: . 
        Use Bootstrap responsive design. Have a clear code. Add code templates for JS for each element. Generate also SEO friendly meta tags. Do not use integrity attribute in link or scripts
        <text: {text} />
        """,
    )
    # chain = SimpleSequentialChain(llm=llm, prompt=prompt,verbose=True)
    chain = LLMChain(llm=llm, prompt=prompt,verbose=True)
    return chain.run(text)

def refine_website_langchain(code,text):
    llm = OpenAIChat(temperature=0.0,model="gpt-3.5-turbo",max_tokens=2900,openai_api_key=st.session_state['OpenAPIKey'])
    prompt = PromptTemplate(
    input_variables=["code","text"],
    template="""
        You will be frontend web developer and output necessary html, css, javascript code. The current website code is given in <code:, user will give next command in <text: . 
        Do not use integrity attribute in link or scripts. Based on the current code, you should regenerate the code according to <text. 
        <code: ```{code}``` />
        <text: ```{text}``` />
        """,
    )
    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    return chain.run({'code':code,'text':text})

# def generate_website(text):
#     openai.api_key  = st.session_state['OpenAPIKey']
#     print(st.session_state['OpenAPIKey'])
#     prompt = f"""
#         You will act as frontend web developer and output necessary html, css, javascript code. User will describe what the web page will contain in delimeter <text: . 
#         You should always chose responsive design. Have a clear code. Add code templates for JS for each element. Generate also SEO friendly meta tags. Do not use integrity attribute in link or scripts
#         <text: ```{text}``` />
#         """
#     print(prompt)
#     return get_completion(prompt,temp=0)

# def refine_website(code,text):
#     openai.api_key  = st.session_state['OpenAPIKey']
#     print(st.session_state['OpenAPIKey'])
#     prompt = f"""
#         You will act as frontend web developer and output necessary html, css, javascript code. The current website code is given in <code:, user will give next command in <text: . 
#         Do not use integrity attribute in link or scripts. Based on the current code, you should regenerate the code according to <text. 
#         <code: ```{code}``` />
#         <text: ```{text}``` />
#         """
#     print(prompt)
#     return get_completion(prompt,temp=0)

def define_nextSteps_2(code,text):
    llm = OpenAIChat(temperature=0.0,model="gpt-3.5-turbo",max_tokens=2900,openai_api_key=st.session_state['OpenAPIKey'])
    prompt = PromptTemplate(
    input_variables=["code","text"],
    template="""
        You will be application developer and output necessary code. The current website code is given in <code:, user will give next command in <text: . 
        Do not use integrity attribute in link or scripts. Based on the current code, you should refactor given code as frontend and also output backend API code according to <text. If a database required, output Create SQL for all database tables
        <code: ```{code}``` />
        <text: ```{text}``` />
        Response should be in following format:
        Frontend Code: <Generated Frontend code>
        Backend Code: <Generated Backend code> 
        Database SQL: <Generated Create SQL>
        """,
    )
    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    response=chain.run({'code':code,'text':text})
    start_index=response.find("Frontend Code:")+len("Frontend Code:")
    end_index=response.find("Backend Code:")
    last_index=response.find("Database SQL:")
    frontend_str=response[start_index:end_index]
    backend_str=response[end_index+len("Backend Code:"):last_index]
    sql_str=response[last_index+len("Database SQL:"):len(response)]
    return [frontend_str.replace("```",""),backend_str.replace("```",""),sql_str.replace("```","")]

def define_nextSteps_1(code):
    openai.api_key  = st.session_state['OpenAPIKey']
    prompt = f"""
        You will act as application developer, Consider that the website requires any dynamic functionality or data retrieval from a server Define what backend API or external app integrations should be done with this code: also possible changes on frontend code, 
        ```{code}```.
        """
    return get_completion(prompt,temp=0)

def get_completion(prompt, model="gpt-3.5-turbo",temp=0):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temp, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

if 'counter' not in st.session_state:
    st.session_state['counter'] = 0
    

def main():
    with st.sidebar:
        openapikey=st.text_input('Open AI API Key')
        if st.button("Set") :
            st.session_state['OpenAPIKey']=openapikey
    
    if(st.session_state['frontend_finished']):
        st.markdown(st.session_state['next_steps'])
        next_steps=st.text_area('Define next steps')
        but1, but2, remaining  = st.columns([1, 3, 5])
        with but1:
            if st.button("Next") :
                code=define_nextSteps_2(st.session_state['current_web_code'],next_steps)
                st.session_state['final_code']=code
                st.session_state['counter'] += 1
                
        with but2:
            if st.button("Finish and Deploy to NUCAL!") :
                st.markdown("Deploying!....")
        if('final_code' in st.session_state):
            st.subheader("Visual View")
            st.components.v1.html(st.session_state['final_code'][0],height=500,scrolling=True)
            st.subheader("Backend Code")
            st_ace(
                        value=st.session_state['final_code'][0],
                        language="html",
                        theme="chaos",
                        keybinding="vscode",
                        font_size=14,
                        tab_size=4,
                        show_gutter=True,
                        wrap=True,
                        auto_update=False,
                        readonly=False,
                        min_lines=45,
                        key=f"ace-2editor-{st.session_state['counter']}",
                    )
            st.subheader("Backend Code")
            st_ace(
                        value=st.session_state['final_code'][1],
                        language="javascript",
                        theme="chaos",
                        keybinding="vscode",
                        font_size=14,
                        tab_size=4,
                        show_gutter=True,
                        wrap=True,
                        auto_update=False,
                        readonly=False,
                        min_lines=45,
                        key=f"ace-1editor-{st.session_state['counter']}",
                    )
            st.subheader("DB Creation")
            st_ace(
                        value=st.session_state['final_code'][2],
                        language="sql",
                        theme="chaos",
                        keybinding="vscode",
                        font_size=14,
                        tab_size=4,
                        show_gutter=True,
                        wrap=True,
                        auto_update=False,
                        readonly=False,
                        min_lines=45,
                        key=f"ace-3editor-{st.session_state['counter']}",
                    )
            print(next_steps)

    else:    
        if('current_web_code' not in st.session_state): #First Site Creation
            website_desc=st.text_area('Define web aplication you want to build','Create a homepage template using Bootstrap responsive design. Include the menu for the homepage, services, portfolio, blog, and about us on the header.')
            if st.button("Generate Website") : 
                # output=generate_website(website_desc)
                output=generate_website_langchain(website_desc)
                st.session_state['current_web_code']=output
                st.session_state['web_code_versions']=[{'comment':website_desc  , 'code':output}]
                print("Generate Website")
                st.experimental_rerun()
        else:
            print("Printing Refine")
            comments = [item['comment'] for item in st.session_state['web_code_versions']]
            print(comments)
            selected_comment = st.selectbox('Select a version to revert!', comments)
            refine_text=st.text_area('Refine current web aplication')
            cR1, cR2, cR3, cR4 = st.columns([1, 1, 1, 3])
            with cR1:
                if st.button("Revert to version!"):
                    selected_code = next(item['code'] for item in st.session_state['web_code_versions'] if item['comment'] == selected_comment)
                    st.session_state['current_web_code']=selected_code
                    
            with cR2:
                if st.button("Regenerate!") :
                    # newoutput=refine_website(st.session_state['current_web_code'],refine_text)
                    newoutput=refine_website_langchain(st.session_state['current_web_code'],refine_text)
                    st.session_state['current_web_code']=newoutput
                    st.session_state['web_code_versions'].append({'comment':refine_text  , 'code':newoutput})
                    st.session_state['counter'] += 1
                    st.experimental_rerun()
            with cR3:
                if st.button("I am done with the frontend, Next:"):
                    st.session_state['frontend_finished'] = True
                    next_steps=define_nextSteps_1(st.session_state['current_web_code'])
                    print(next_steps)
                    st.session_state['next_steps']=next_steps
                    st.experimental_rerun()
            st.text("Here is current website you created")
            st.components.v1.html(st.session_state['current_web_code'],height=500,scrolling=True)
            
            c1, c2 = st.columns([3, 1])

            c2.subheader("Parameters")

            with c1:
                print("Printing Editor",st.session_state['current_web_code'])
                st.session_state['current_web_code'] = st_ace(
                    value=st.session_state['current_web_code'],
                    language=c2.selectbox("Language mode", ["html","vue","javascript"], index=0),
                    theme=c2.selectbox("Theme", options=THEMES, index=1),
                    keybinding=c2.selectbox("Keybinding mode", options=KEYBINDINGS, index=3),
                    font_size=c2.slider("Font size", 5, 24, 14),
                    tab_size=c2.slider("Tab size", 1, 8, 4),
                    show_gutter=c2.checkbox("Show gutter", value=True),
                    wrap=c2.checkbox("Wrap enabled", value=True),
                    auto_update=c2.checkbox("Auto update", value=False),
                    readonly=c2.checkbox("Read-only", value=False),
                    min_lines=45,
                    key=f"ace-editor-{st.session_state['counter']}",
                )
        

if __name__ == "__main__":
    st.set_page_config(page_title="AI Web Application Generator", layout="wide")
    st.header("AI Web Application Generator")
    hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
    st.markdown(hide_default_format, unsafe_allow_html=True)
    main()