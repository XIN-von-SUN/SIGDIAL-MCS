import telegram
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, InputMediaVideo

from telegram.ext import Updater, MessageHandler, Filters, CallbackQueryHandler
from telegram.ext import CommandHandler

# from bot_model import get_info
from post import post_user, get_response_tracker
from requests import *

telegram_bot_token = "5153556322:AAGkD06LGXRCUrEGsG6fEgV9YQlmlVMTzyU"

updater = Updater(token=telegram_bot_token, use_context=True)
dispatcher = updater.dispatcher


# # set up the introductory statement for the bot when the /start command is invoked
# def start(update, context):
# chat_id = update.effective_chat.id
# print(f'chat_id is: {chat_id}')
#     context.bot.send_message(chat_id=update.effective_chat.id, text="Hello there!")


# obtain the user input and send it to bot model and get response from bot
def BotInteraction(update, context):
    """
    1. can handle pure text iput considering dialogue trackers;     solved
    2. can handle command input;      solved (just ignore all command message and treat them as the free text)
    3. can handle button in response;      solved
    4. can handle url attachment;
    """
    url_bot_post = "http://a936-145-38-196-223.ngrok.io/webhooks/rest/webhook"
    user_utter = update.message.text
    sender_id = update.effective_chat.id
    data_json = {"sender": sender_id, "message": user_utter}
    bot_response = post_user(sender_id, url_bot_post, user_utter, data_json)
    print(f"user_utter is: {user_utter}\nbot_response is: {bot_response}\n")

    # # get bot tracker information
    # url_bot_tracker = f"http://9a80-145-38-196-223.ngrok.io/conversations/{sender_id}/tracker"
    # tracker = get_response_tracker(url_bot_tracker)
    # # print(f'tracker is: {tracker}\n')

    # to handle buttons in response
    for reply in bot_response:
        if "buttons" in reply.keys():
            buttons = []
            for butt in reply["buttons"]:
                buttons.append(
                    [
                        InlineKeyboardButton(
                            text=butt["title"], callback_data=butt["payload"]
                        )
                    ]
                )
            reply_markup = InlineKeyboardMarkup(buttons)
            update.message.reply_text(text=reply["text"], reply_markup=reply_markup)
        elif "attachment" in reply.keys():
            attachment_content = get(reply["attachment"]["payload"]).content
            # print(f'attachment_content is: {attachment_content}')
            context.bot.send_video(
                chat_id=update.effective_chat.id, video=attachment_content
            )
        else:
            update.message.reply_text(text=reply["text"])

    """
    Test
    """
    # update.message.reply_text("Beginning of inline keyboard")
    # bot_response = [{'recipient_id': '1935537105', 'text': "Let's go to the video!"}, {'recipient_id': '1935537105', 'attachment': {'payload': 'https://www.youtube.com/watch?v=rBUjOY12gJA&t=172s'}}, {'recipient_id': '1935537105', 'text': 'Seems we can move on to the next topic!'}, {'recipient_id': '1935537105', 'text': 'Shall we talk about pa?', 'buttons': [{'title': 'Sure', 'payload': '/trigger_pa'}, {'title': "No, I don't want to.", 'payload': '/deny'}]}]
    # # https://www.youtube.com/watch?v=rBUjOY12gJA
    # #[{'recipient_id': '1935537105', 'text': "Nice! Let's talk about the role of physical activity in your personal life, and I'm going to ask you a couple of questions. Does that sound good to you?", 'buttons': [{'title': 'Sure!', 'payload': '/trigger_pa'}, {'title': "No, I don't want to", 'payload': '/deny'}]}]

    # # to handle buttons in response
    # for reply in bot_response:
    #     if 'buttons' in reply.keys():
    #         buttons = []
    #         for butt in reply['buttons']:
    #             buttons.append([InlineKeyboardButton(text=butt['title'], callback_data=butt['payload'])])
    #         reply_markup = InlineKeyboardMarkup(buttons)
    #         update.message.reply_text(text=reply['text'], reply_markup=reply_markup)
    #     elif 'attachment' in reply.keys():
    #         attachment_content = get(reply['attachment']['payload']).content
    #         # print(f'attachment_content is: {attachment_content}')
    #         context.bot.send_video(chat_id=update.effective_chat.id, video=attachment_content)
    #     else:
    #         update.message.reply_text(text=reply['text'])


def ButtonQueryHandler(update, context):
    query = update.callback_query.data
    # update.callback_query.answer()  # output is True/False
    # print(f'button callback data is: {query}\n')

    url_bot_post = "http://a936-145-38-196-223.ngrok.io/webhooks/rest/webhook"
    user_utter = query
    sender_id = update.effective_chat.id
    data_json = {"sender": sender_id, "message": user_utter}
    bot_response = post_user(sender_id, url_bot_post, user_utter, data_json)
    print(f"user_utter is: {user_utter}\nbot_response is: {bot_response}\n")

    for reply in bot_response:
        if "buttons" in reply.keys():
            buttons = []
            for butt in reply["buttons"]:
                buttons.append(
                    [
                        InlineKeyboardButton(
                            text=butt["title"], callback_data=butt["payload"]
                        )
                    ]
                )
            reply_markup = InlineKeyboardMarkup(buttons)
            context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=reply["text"],
                reply_markup=reply_markup,
            )
        elif "attachment" in reply.keys():
            attachment_content = get(reply["attachment"]["payload"]).content
            # print(f'attachment_content is: {attachment_content}')
            context.bot.send_video(
                chat_id=update.effective_chat.id, video=attachment_content
            )
        else:
            context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=reply["text"]
            )

    """
    Test
    """
    # global response_button
    # if query == '/trigger_pa':
    #     response_button = [{'recipient_id': '1935537105', 'text': "Wow",
    #                     'buttons': [{'title': 'Sure!', 'payload': 'yes'}, {'title': "No, I don't want to", 'payload': 'no'}]}]

    #     for reply in response_button:
    #         if 'buttons' in reply.keys():
    #             buttons = []
    #             for butt in reply['buttons']:
    #                 buttons.append([InlineKeyboardButton(text=butt['title'], callback_data=butt['payload'])])
    #             reply_markup = InlineKeyboardMarkup(buttons)
    #             context.bot.send_message(chat_id=update.effective_chat.id, text=reply['text'], reply_markup=reply_markup)
    #         elif 'attachment' in reply.keys():
    #             attachment_content = get(reply['attachment']['payload']).content
    #             # print(f'attachment_content is: {attachment_content}')
    #             context.bot.send_video(chat_id=update.effective_chat.id, video=attachment_content)
    #         else:
    #             context.bot.send_message(chat_id=update.effective_chat.id, text=reply['text'], reply_markup=reply_markup)
    # elif query == '/deny':
    #     context.bot.send_message(chat_id=update.effective_chat.id, text="deny input!") #update.message.reply_text("deny input")


# # invoke function when the user send a command message
# dispatcher.add_handler(CommandHandler("start", start))
# dispatcher.add_handler(CommandHandler("stop", stop))

# invoke function when the user sends a message which is not a command.
dispatcher.add_handler(MessageHandler(Filters.text, BotInteraction))

# invoke function when the user click a button and will get callback data which is not a command.
dispatcher.add_handler(CallbackQueryHandler(ButtonQueryHandler))

updater.start_polling()
