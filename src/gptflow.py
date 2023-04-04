import openai

class MemoryStore:
  def __init__(self):
    self.store = None

  def get(self):
    return self.store

  def put(self, value):
    self.store = value

class GptFlow:
  def __init__(self, apiKey, stateStore=MemoryStore):
    openai.api_key = apiKey
    self.stateStore = stateStore()
    self.start_sequence = "\nAI:"
    self.restart_sequence = "\nHuman: "

  def __compose_prompt__(self, state, input):
    setup_prompt = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n"
    dialog_prompt = ""
    for qa in state:
        dialog_prompt += self.restart_sequence + qa[0] + self.start_sequence + qa[1]
    dialog_prompt += self.restart_sequence + input
    return setup_prompt + dialog_prompt

  
  def clear_state(self):
    self.stateStore.put(None)

  def interact(self, input):
    state = self.stateStore.get()
    if state is None:
        state = list()
    # Call interactions
    prompt = self.__compose_prompt__(state, input)
    print(prompt)
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.9,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stop=[" Human:", " AI:"]
        )
    reply = response["choices"][0]["text"]
    print(reply)
    # Save state
    state.append((input, reply))
    self.stateStore.put(state)

    # Return response
    return response
