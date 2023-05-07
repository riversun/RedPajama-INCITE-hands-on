class ChatContent:
    def __init__(self, role: str, msg: str = ""):
        self.role = role
        self.message = msg

    def get_role(self):
        return self.role

    def get_message(self):
        return self.message


class ChatPrompt:
    """
    A builder to build chat prompts according to the characteristics of each language model.
    """

    def __init__(self):
        self.system = ""
        self.chat_contents = []
        self.responder_messages = []
        self.requester_messages = []
        self.requester = ""
        self.responder = ""

    def set_system(self, system):
        """
        Set initial prompts for "system."
        :param system:
        :return:
        """
        self.system = system

    def set_requester(self, requester):
        """
        Sets the role name of the requester (=user)
        :param requester:
        :return:
        """
        self.requester = requester

    def set_responder(self, responder):
        """
        Sets the role name of the responder (=AI)
        :param responder:
        :return:
        """
        self.responder = responder

    def add_requester_msg(self, message):
        self._add_msg(ChatContent(role=self.requester, msg=message))

    def add_responder_msg(self, message):
        self._add_msg(ChatContent(role=self.responder, msg=message))

    def set_responder_last_msg(self, message):
        self.responder_messages[-1].message = message

    def get_requester_last_msg(self):
        """
        Retrieve the latest message from the requester
        :return:
        """
        return self.requester_messages[-1].message

    def _add_msg(self, msg):
        self.chat_contents.append(msg)
        if msg.role == self.responder:
            self.responder_messages.append(msg)
        elif msg.role == self.requester:
            self.requester_messages.append(msg)

    def is_requester_role(self, role):
        if self.requester == role:
            return True
        else:
            return False

    def get_skip_len(self):
        """
        （Get the length to skip (already entered as a prompt)
        :return:
        """
        current_prompt = self.create_prompt()

        skip_echo_len = len(current_prompt)
        return skip_echo_len

    def get_stop_strs(self):
        return [
            '<|endoftext|>',
            '\n<'
            # Safety stop valve when the model generates not only AI conversations but also human parts of the conversation.
        ]

    def create_prompt(self):
        """
        Build prompts according to the characteristics of each language model
        :return:
        """
        ret = self.system;
        for chat_content in self.chat_contents:
            chat_content_role = chat_content.get_role()
            chat_content_message = chat_content.get_message()
            if chat_content_role:
                if chat_content_message:
                    merged_message = chat_content_role + ": " + chat_content_message + "\n"
                else:
                    merged_message = chat_content_role + ":"
                ret += merged_message

        return ret


# portable UT
if False:
    chatPrompt = ChatPrompt()

    chatPrompt.set_requester("<human>")
    chatPrompt.set_responder("<bot>")
    chatPrompt.add_requester_msg("Who is Alan Turing")
    chatPrompt.add_responder_msg(None)

    assert """<human>: Who is Alan Turing
<bot>:""" == chatPrompt.create_prompt()
