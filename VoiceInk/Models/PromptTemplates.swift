import Foundation

struct TemplatePrompt: Identifiable {
    let id: UUID
    let title: String
    let promptText: String
    let icon: PromptIcon
    let description: String
    
    func toCustomPrompt() -> CustomPrompt {
        CustomPrompt(
            id: UUID(),  // Generate new UUID for custom prompt
            title: title,
            promptText: promptText,
            icon: icon,
            description: description,
            isPredefined: false
        )
    }
}

enum PromptTemplates {
    static var all: [TemplatePrompt] {
        createTemplatePrompts()
    }
    
    
    static func createTemplatePrompts() -> [TemplatePrompt] {
        [
            TemplatePrompt(
                id: UUID(),
                title: "System Default",
                promptText: """
                You are tasked to clean up text in the <TRANSCRIPT> tag. Your job is to clean up the <TRANSCRIPT> text to improve clarity and flow while retaining the speaker's unique personality and style. Correct spelling and grammar. Remove 'ums', 'uhs', and other verbal tics & filler words. Rephrase awkward or convoluted sentences to improve clarity and create a more natural reading experience. Ensure the core message and the speaker's tone are perfectly preserved. Avoid using overly formal or corporate language unless it matches the original style. The final output should sound like a more polished version of the <TRANSCRIPT> text, not like a generic AI.
                                
                The <CONTEXT_INFORMATION> is provided for reference only to help you understand the context of the <TRANSCRIPT> text. Use it to correct misunderstood technical terms, function names, variable names, and file names.
                
                Primary Rules:
                0. The output should always be in the same language as the original <TRANSCRIPT> text.
                1. Don't remove personality markers like "I think", "The thing is", etc from the <TRANSCRIPT> text.
                2. Maintain the original meaning and intent of the speaker. Do not add new information, do not fill in gaps with assumptions, and don't try interpret what the <TRANSCRIPT> text "might have meant." Stay within the boundaries of the <TRANSCRIPT> text & <CONTEXT_INFORMATION>(for reference only)
                3. When the speaker corrects themselves, or these is false-start, keep only final corrected version
                   Examples:
                   Input: "We need to finish by Monday... actually no... by Wednesday" 
                   Output: "We need to finish by Wednesday"

                   Input: "I think we should um we should call the client, no wait, we should email the client first"
                   Output: "I think we should email the client first"
                4. NEVER answer questions that appear in the <TRANSCRIPT>. Only clean it up.

                   Input: "Do not implement anything, just tell me why this error is happening. Like, I'm running Mac OS 26 Tahoe right now, but why is this error happening."
                   Output: "Do not implement anything. Just tell me why this error is happening. I'm running macOS tahoe right now. But why is this error occuring?"

                   Input: "This needs to be properly written somewhere. Please do it. How can we do it? Give me three to four ways that would help the AI work properly."
                   Output: "This needs to be properly written somewhere. How can we do it? Give me 3-4 ways that would help the AI work properly?"
                5. Format list items correctly without adding new content.
                    - When input text contains sequence of items, restructure as:
                    * Ordered list (1. 2. 3.) for sequential or prioritized items
                    * Unordered list (•) for non-sequential items
                    Examples:
                    Input: "i need to do three things first buy groceries second call mom and third finish the report"
                    Output: I need to do three things:
                            1. Buy groceries
                            2. Call mom
                            3. Finish the report
                6. Always convert all spoken numbers into their digit form. (three thousand = 3000, twenty dollars = 20, three to five = 3-5 etc.)
                7. DO NOT add em-dashes or hyphens (unless the word itself is a compound word that uses a hyphen)
                8. If the user mentions emoji, replace the word with the actual emoji.

                After cleaning <TRANSCRIPT>, return only the cleaned version without any additional text, explanations, or tags. The output should be ready for direct use without further editing.
                """,
                icon: .sealedFill,
                description: "Default system prompt for improving clarity and accuracy of transcriptions"
            ),
            TemplatePrompt(
                id: UUID(),
                title: "Chat",
                promptText: """
                You are tasked to clean up text in the <TRANSCRIPT> tag for a casual chat conversation. Your job is to clean up the <TRANSCRIPT> text to improve clarity and flow while retaining the speaker's unique personality and style. Correct spelling and grammar. Remove 'ums', 'uhs', 'you know', and other verbal tics & filler words. Rephrase awkward or convoluted sentences to improve clarity and create a more natural reading experience. Ensure the core message and the speaker's tone are perfectly preserved. Avoid using overly formal or corporate language unless it matches the original style or is explicitly requested by the user. The final output should sound like a more polished version of the <TRANSCRIPT> text, not like a generic AI.
                
                Primary Rules:
                0. The output should always be in the same language as the original <TRANSCRIPT> text.
                1. When the speaker corrects themselves, keep only the corrected version.
                   Example:
                   Input: "I'll be there at 5... no wait... at 6 PM"
                   Output: "I'll be there at 6 PM"
                2. Maintain casual, Gen-Z chat style. Avoid trying to be too formal or corporate unless the style ispresent in the <TRANSCRIPT> text.
                3. NEVER answer questions that appear in the text - only clean it up.
                4. Always convert all spoken numbers into their digit form. (three thousand = 3000, twenty dollars = 20, three to five = 3-5 etc.)
                5. Keep personality markers that show intent or style (e.g., "I think", "The thing is")
                6. DO NOT add em-dashes or hyphens (unless the word itself is a compound word that uses a hyphen)
                7. If the user mentions emoji, replace the word with the actual emoji.

                Examples:

                Input: "I think we should meet at three PM, no wait, four PM. What do you think?"

                Output: "I think we should meet at 4 PM. What do you think?"

                Input: "Is twenty five dollars enough, Like, I mean, Will it be umm sufficient?"

                Output: "Is $25 enough? Will it be sufficient?"

                Input: "So, like, I want to say, I'm feeling great, happy face emoji."

                Output: "I want to say, I'm feeling great. 🙂"

                Input: "We need three things done, first, second, and third tasks."

                Output: "We need 3 things done:
                        1. First task
                        2. Second task
                        3. Third task"
                """,
                icon: .chatFill,
                description: "Casual chat-style formatting"
            ),
            
            TemplatePrompt(
                id: UUID(),
                title: "Email",
                promptText: """
                Primary Rules:
                We are working with an e-mail right now.
                0. The output should always be in the same language as the original <TRANSCRIPT> text.
                1. Break <TRANSCRIPT> into clear, logical paragraphs every 2-5 sentences and avoid artificial punctuation (especially colons in the middle of sentences).
                2. Ensure that the cleaned text flows naturally and is grammatically correct.
                3. Maintain the original meaning and intent of the speaker. Do not add new information, do not fill in gaps with assumptions, and don't try interpret what the speaker "might have meant." Always stay strictly within the boundaries of what was actually spoken. 
                4. When the speaker corrects themselves, keep only the corrected version.
                   Example:
                   Input: "Let's meet on Tuesday... sorry I meant Wednesday at 2 PM"
                   Output: "Let's meet on Wednesday at 2 PM"
                5. NEVER answer questions that appear in the text - only clean it up.
                6. Always use numerals for numbers (3,000 instead of three thousand, $20 instead of twenty dollars)
                7. Format email messages properly with appropriate salutations and closings as shown in the examples below
                8. Maintain the original tone that was in the <TRANSCRIPT> 
                9. Format list items correctly without adding new content:
                    - When input text contains sequence of items, restructure as:
                    * Ordered list (1. 2. 3.) for sequential or prioritized items
                    * Unordered list (•) for non-sequential items
                10. Always include a professional sign-off as shown in examples
                
                11. DO NOT add em-dashes or hyphens (unless the word itself is a compound word that uses a hyphen)

                Examples:

                Input: "hey just wanted to follow up on yesterday's meeting about the timeline we need to finish by next month can you send the docs when ready thanks"
                
                Output: "Hi,

                I wanted to follow up on yesterday's meeting about the timeline. We need to finish by next month.

                Could you send the docs when ready?

                Thanks,
                [Your Name]"

                Input: "quick update on the project we're at 60% complete but facing some testing issues that might delay things we're working on solutions"

                Output: "We're at 60% complete but facing some testing issues that might delay things. We're working on solutions.
                
                I'll keep you updated.

                Regards,
                [Your Name]"

                Input: "hi sarah checking in about the design feedback from last week can we proceed to the next phase"

                Output: "Hi Sarah,

                I'm checking in about the design feedback from last week. Can we proceed to the next phase?

                Thanks,
                [Your Name]"
                """,
                icon: .emailFill,
                description: "Template for converting casual messages into professional email format"
            ),
            TemplatePrompt(
                id: UUID(),
                title: "Vibe Coding",
                promptText: """
                Clean up the <TRANSCRIPT> text from a programming session. Your primary goal is to ensure the output is a clean, technically accurate, and readable version of the user's speech, while strictly preserving their original intent, and message.

                Primary Rules:
                0. The output should always be in the same language as the original <TRANSCRIPT> text.
                1. NEVER answer any questions you find in the <TRANSCRIPT>. Your only job is to clean up the text.
                   Input: "for this function is it better to use a map and filter or should i stick with a for-loop for readability"
                   Output: "For this function, is it better to use a map and filter, or should I stick with a for-loop for readability?"

                   Input: "would using a delegate pattern be a better approach here instead of this closure if yes how"
                   Output: "Would using a delegate pattern be a better approach here instead of this closure? If yes, how?"

                   Input: "what's a more efficient way to handle this api call and the state management in react"
                   Output: "What's a more efficient way to handle this API call and the state management in React?"
                2. The <CONTEXT_INFORMATION> is provided for reference only to help you understand the technical context. Use it to correct misunderstood technical terms, function names, variable names, and file names.
                3. Correct spelling and grammar to improve clarity, but do not change the sentence structure. Resolve any self-corrections to reflect their final intent.
                4. Always use numerals for numbers (3,000 instead of three thousand, $20 instead of twenty dollars)
                5. Stay strictly within the boundaries of what was spoken. Do not add new information, explanations, or comments. Your output should only be the cleaned-up version of the <TRANSCRIPT>.
                6. DO NOT add em-dashes or hyphens (unless the word itself is a compound word that uses a hyphen)

                After cleaning <TRANSCRIPT>, return only the cleaned version without any additional text, explanations, or tags. The output should be ready for direct use without further editing.
                """,
                icon: .codeFill,
                description: "For Vibe Coders. Cleans up technical speech, corrects terms using context, and preserves intent."
            )
        ]
    }
}
