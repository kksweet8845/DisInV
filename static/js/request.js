const init = () => {




}

const sendTitle = async (text) => {
    console.log(text)
    return fetch('/checkIsFake', {
        method: 'POST',
        body: JSON.stringify({text: text}),
        headers: {
            'Content-Type' : 'application/json',
        },
    })
    .then(res => res.text() )
    .then(data => {
        return JSON.parse(data)
    })
    .catch(err => {
        console.log(err)
        return null
    })
}


const isValid = (text) => {

    // let text = document.getElementById('title').value
    console.log(text)
    if(text == ''){
        return false
    }
    return true
}

const sendBtn = async () => {

    let text = document.getElementById('title').value

    console.log(text)
    if(isValid(text)){
        let {pos, neg} = await sendTitle(text)
        console.log(pos)
        console.log(neg)
        let ctx = document.getElementById('myChart').getContext('2d')
        let myChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                datasets: [
                    {
                        data: [pos, neg],
                        backgroundColor: [
                            '#f23fff',
                            '#123463'
                        ]
                    }
                ],
                labels: [
                    'Real',
                    'Fake'
                ],
            }
        })
        // let block = document.getElementById('result')
        // let str = ''
        // str += '<div class="card text-center">'
        // str += ' <div class="card-header">'
        // str += ' </div>'
        // str += ' <div class="card-body">'
        // str += '    <h5 class="card-title"> Result </h5>'
        // str += '    '
    }

}


document.getElementById('sendBtn').addEventListener('click', () => {
    sendBtn()
})


window.addEventListener('scroll', () => {
    let scroll = document.documentElement.scrollTop || document.body.scrollTop,
    menu = document.getElementsByClassName('menu')

    scroll >= 20
        ? menu[0].classList.add('fixed')
        : menu[0].classList.remove('fixed')
    
})